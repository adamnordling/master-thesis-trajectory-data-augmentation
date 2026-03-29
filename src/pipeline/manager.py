import logging
import os
from collections.abc import Callable
from multiprocessing import Pool, cpu_count
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.evaluation.analyze import main as run_report_main
from src.evaluation.models import train_and_evaluate
from src.evaluation.reporting import create_final_summary_reports

# --- Internal Imports ---
from src.pipeline.workers import (
    augment_data_worker,
    extract_aug_features_worker,
    extract_features_worker,
    prepare_dataset_worker,
)
from src.utils import load_dataframe

# Initialize Logger
logger = logging.getLogger(__name__)


class PipelineManager:
    """Orchestrates the data processing pipeline.

    Responsibilities:
    1. Discovery of datasets.
    2. Management of parallel worker processes.
    3. Sequencing of pipeline steps (Prepare -> Extract -> Augment -> Eval).
    4. Managing the complex output directory structure for results.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initializes the PipelineManager with configuration settings."""
        self.config = config

        # 1. Settings
        self.seeds: list[int] = config["seeds"]
        self.model_config: dict[str, Any] = config.get("model_config", {})
        self.use_parallel: bool = config["processing"]["parallel"]
        self.cpu_limit: float = config["processing"]["cpu_usage_limit"]

        # 2. Base Paths
        root = config["paths"]["project_root"]
        self.raw_dir = os.path.join(root, config["paths"]["raw_data"])
        self.aug_dir = os.path.join(root, config["paths"]["aug_data"])
        self.output_root = os.path.join(root, config["paths"]["results"])

        # 3. Define Professional Output Structure
        self.dirs = {
            "states": os.path.join(self.output_root, "model_states"),
            "opt_history": os.path.join(self.output_root, "optimization", "history"),
            "opt_details": os.path.join(self.output_root, "optimization", "details"),
            "analysis": os.path.join(self.output_root, "analysis"),
        }

    def _get_max_workers(self) -> int:
        """Calculates safe number of worker processes based on config."""
        try:
            import psutil

            physical_cores = psutil.cpu_count(logical=False)
            if physical_cores is None:
                physical_cores = cpu_count()
        except (ImportError, AttributeError):
            physical_cores = cpu_count()
        return max(1, int(physical_cores * self.cpu_limit))

    def _discover_raw_datasets(self) -> dict[str, dict[str, str]]:
        """Scans the data/raw directory for CSV files."""
        datasets: dict[str, dict[str, str]] = {}
        if not os.path.exists(self.raw_dir):
            logger.error(f"Raw data directory not found: {self.raw_dir}")
            return datasets

        for filename in os.listdir(self.raw_dir):
            if filename.endswith(".csv"):
                dataset_name = os.path.splitext(filename)[0]
                datasets[dataset_name] = {"name": dataset_name, "csv_file": filename}
        return datasets

    def _run_tasks(self, worker_func: Callable, tasks: list[Any], desc: str) -> None:
        """Generic driver for parallel execution."""
        if not tasks:
            logger.info(f"No tasks to run for: {desc}")
            return

        results: list[dict[str, Any]] = []
        if self.use_parallel and len(tasks) > 1:
            n_workers = self._get_max_workers()
            n_workers = min(n_workers, len(tasks))

            logger.info(f"Running '{desc}' with {len(tasks)} tasks on {n_workers} cores...")

            with Pool(processes=n_workers) as pool:
                results = list(tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks), desc=desc))
        else:
            logger.info(f"Running '{desc}' sequentially...")
            results = [worker_func(t) for t in tqdm(tasks, desc=desc)]

        errors = [res for res in results if not res.get("success", False)]
        if errors:
            logger.error(f"!!! Encountered {len(errors)} errors during '{desc}' !!!")
            for i, e in enumerate(errors):
                logger.error(f"  [Error {i + 1}] Task: {e.get('task')} -> Message: {e.get('error')}")

    def run_preparation(self, datasets: list[str]) -> None:
        """Step 1: Data Preparation.
        Raw CSV -> Stratified Split Feather files in data/augmented/{dataset}/{seed}/.
        """
        available_raw = self._discover_raw_datasets()
        tasks = []

        for name in datasets:
            if name not in available_raw:
                logger.warning(f"Dataset '{name}' not found in {self.raw_dir}")
                continue

            for seed in self.seeds:
                target_file = os.path.join(self.aug_dir, name, str(seed), f"train_seed_{seed}_pts.feather")
                if not os.path.exists(target_file):
                    tasks.append((seed, name, available_raw[name]))

        self._run_tasks(prepare_dataset_worker, tasks, "Preparing Datasets")

    def run_feature_extraction(self, datasets: list[str]) -> None:
        """Step 2: Base Feature Extraction.
        Calculates features for the standard Train and Test sets.
        """
        tasks: list[str] = []
        for name in datasets:
            for seed in self.seeds:
                seed_dir = os.path.join(self.aug_dir, name, str(seed))
                if not os.path.exists(seed_dir):
                    continue

                for split in ["train", "test"]:
                    pts_file = os.path.join(seed_dir, f"{split}_seed_{seed}_pts.feather")
                    target_file = os.path.join(seed_dir, f"{split}_seed_{seed}_pts_trajectory_features.feather")

                    if os.path.exists(pts_file) and not os.path.exists(target_file):
                        tasks.append(pts_file)

        self._run_tasks(extract_features_worker, tasks, "Extracting Raw Features")

    def run_augmentation(
        self,
        datasets: list[str],
        strategies: list[str],
        proportion: float,
        n_aug_trajs: int,
        points_proportion: float,
        run_suffix: str = "",
    ) -> None:
        """Step 3: Augmentation.
        Generates new trajectories based on the selected strategies.
        """
        tasks = []
        for name in datasets:
            for seed in self.seeds:
                for strategy in strategies:
                    target_file = os.path.join(
                        self.aug_dir,
                        name,
                        str(seed),
                        f"train_seed_{seed}_pts_augmented_{strategy}{run_suffix}.feather",
                    )

                    if not os.path.exists(target_file):
                        tasks.append((seed, strategy, proportion, n_aug_trajs, points_proportion, name, run_suffix))

        self._run_tasks(augment_data_worker, tasks, f"Augmenting Data {run_suffix}")

    def run_aug_feature_extraction(self, datasets: list[str], strategies: list[str], run_suffix: str = "") -> None:
        """Step 4: Augmented Feature Extraction.
        Calculates features for the new data and merges with original features.
        """
        tasks = []
        for name in datasets:
            for seed in self.seeds:
                for strategy in strategies:
                    aug_file = os.path.join(
                        self.aug_dir,
                        name,
                        str(seed),
                        f"train_seed_{seed}_pts_augmented_{strategy}{run_suffix}.feather",
                    )
                    target_file = os.path.join(
                        self.aug_dir,
                        name,
                        str(seed),
                        f"train_seed_{seed}_pts_trajectory_features_merged_{strategy}{run_suffix}.feather",
                    )

                    if os.path.exists(aug_file) and not os.path.exists(target_file):
                        tasks.append((seed, strategy, name, run_suffix))

        self._run_tasks(extract_aug_features_worker, tasks, f"Extracting Aug Features {run_suffix}")

    def run_evaluation(self, datasets: list[str], strategies: list[str], run_suffix: str = "") -> str | None:
        """Step 5: Training and Evaluation."""
        all_results_dfs: list[pd.DataFrame] = []

        os.makedirs(self.dirs["states"], exist_ok=True)
        os.makedirs(self.dirs["opt_history"], exist_ok=True)
        os.makedirs(self.dirs["opt_details"], exist_ok=True)

        safe_jobs = self._get_max_workers()

        for dataset in datasets:
            logger.info(f"Starting evaluation for dataset: {dataset}")
            dataset_dir = os.path.join(self.aug_dir, dataset)
            details_dir = os.path.join(self.dirs["opt_details"], dataset)
            os.makedirs(details_dir, exist_ok=True)
            output_csv = os.path.join(details_dir, f"{dataset}{run_suffix}.csv")

            params_cache_path = os.path.join(self.dirs["states"], f"{dataset}_best_params.csv")

            if os.path.exists(params_cache_path):
                cached_params = pd.read_csv(params_cache_path)
            else:
                cached_params = pd.DataFrame(columns=["feature_type", "seed", "model", "best_params"])

            dataset_results: list[dict[str, Any]] = []
            new_params_list: list[dict[str, Any]] = []

            feature_types = ["trajectory_features"] + [
                f"trajectory_features_merged_{s}{run_suffix}" for s in strategies
            ]

            for seed in tqdm(self.seeds, desc=f"Evaluating {dataset}"):
                seed_path = os.path.join(dataset_dir, str(seed))
                if not os.path.exists(seed_path):
                    continue

                test_file = os.path.join(seed_path, f"test_seed_{seed}_pts_trajectory_features.feather")
                if not os.path.exists(test_file):
                    continue

                try:
                    test_df = load_dataframe(test_file)
                except Exception as e:
                    logger.error(f"Failed to load test file: {e}")
                    continue

                for f_name in feature_types:
                    train_file = os.path.join(seed_path, f"train_seed_{seed}_pts_{f_name}.feather")
                    if not os.path.exists(train_file):
                        continue

                    try:
                        train_df = load_dataframe(train_file)
                        results, new_params = train_and_evaluate(
                            train_df=train_df,
                            test_df=test_df,
                            seed=seed,
                            feature_type_name=f_name,
                            cached_params=cached_params,
                            model_config=self.model_config,
                            n_jobs_gridsearch=safe_jobs,
                        )

                        for r in results:
                            r["dataset"] = dataset

                        dataset_results.extend(results)
                        new_params_list.extend(new_params)

                    except Exception as e:
                        logger.error(f"Error evaluating {f_name} seed {seed}: {e}")
                        continue

            if new_params_list:
                new_params_df = pd.DataFrame(new_params_list)
                updated_cache = pd.concat([cached_params, new_params_df], ignore_index=True)
                updated_cache.drop_duplicates(subset=["feature_type", "seed", "model"], keep="last", inplace=True)
                updated_cache.to_csv(params_cache_path, index=False)

            if dataset_results:
                res_df = pd.DataFrame(dataset_results)
                res_df.to_csv(output_csv, index=False)
                all_results_dfs.append(res_df)

        if all_results_dfs:
            combined_df = pd.concat(all_results_dfs, ignore_index=True)
            dataset_prefix = "-".join(sorted(datasets))
            combined_path = os.path.join(self.dirs["opt_history"], f"{dataset_prefix}{run_suffix}.csv")
            combined_df.to_csv(combined_path, index=False)
            return combined_path

        return None

    def generate_reports(self, datasets: list[str]) -> None:
        """Step 6: Generate final reports."""
        create_final_summary_reports(self.output_root, datasets)

    def run_final_analysis(self, datasets: list[str]) -> None:
        """Step 7: Runs the final, consolidated analysis and reporting script."""
        logger.info("Starting final report generation and statistical analysis...")
        for dataset in datasets:
            try:
                logger.info(f"--- Generating final report for dataset: {dataset} ---")
                run_report_main(dataset)
                logger.info(f"--- Successfully generated report for {dataset} ---")
            except Exception as e:
                logger.error(f"Failed to generate final report for {dataset}: {e}", exc_info=True)
