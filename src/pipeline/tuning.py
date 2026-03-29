import logging
import os
from collections.abc import Sequence

import optuna
import pandas as pd

# We type hint using string forward reference or 'Any' to avoid circular imports at runtime
# if manager imports tuning.
from src.pipeline.manager import PipelineManager

logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, manager: PipelineManager, datasets: list[str], strategies: list[str]) -> float:
    """The Optuna Objective Function.

    1. Suggests hyperparameters (Proportion, N_Augmentations, Points_Proportion).
    2. Checks if this combination already exists in the results folder (Caching).
    3. If missing, instructs the Manager to run the specific pipeline steps.
    4. Returns the average F1 score across all datasets for this configuration.
    """
    # 1. Define Search Space
    # We use categorical to allow GridSampler to work effectively
    p: float = trial.suggest_categorical("proportion", [0.1, 0.2, 0.4])
    n: int = trial.suggest_categorical("n_aug_trajs", [1, 3, 5])
    pp: float = trial.suggest_categorical("points_proportion", [0.1, 0.2, 0.4])

    # Create the unique suffix for filenames (e.g., _p20_n3_pp20)
    run_suffix = f"_p{int(p * 100)}_n{n}_pp{int(pp * 100)}"

    logger.info(f"--- Optuna Trial #{trial.number}: Params{run_suffix} ---")

    # 2. Check Global Cache (Combined Results)
    dataset_prefix = "-".join(sorted(datasets))
    combined_csv_path = os.path.join(
        manager.dirs["opt_history"],
        f"{dataset_prefix}{run_suffix}.csv",
    )

    if os.path.exists(combined_csv_path):
        try:
            results_df = pd.read_csv(combined_csv_path)
            # Only skip if the cache contains all the strategies we are looking for!
            existing_strategies = results_df["strategy"].unique()
            if all(s in existing_strategies for s in strategies):
                logger.info(f"Found complete cached results for {run_suffix}. Skipping.")
                aug_res = results_df[results_df["feature_type"] != "trajectory_features"]
                return float(aug_res["score"].mean())
            else:
                logger.warning(f"Cached results for {run_suffix} are incomplete. Re-running.")
            # ----------------------
        except Exception as e:
            logger.warning(f"Cache file corrupt: {e}. Re-running.")

    # 3. Granular Check: Which datasets are missing?
    missing_datasets = []
    cached_dfs = []

    for ds in datasets:
        ds_csv = os.path.join(manager.dirs["opt_details"], ds, f"{ds}{run_suffix}.csv")
        if os.path.exists(ds_csv):
            try:
                cached_dfs.append(pd.read_csv(ds_csv))
                logger.debug(f"Loaded cache for {ds}")
            except Exception:
                missing_datasets.append(ds)
        else:
            missing_datasets.append(ds)

    # 4. Execute Pipeline for Missing Data
    if missing_datasets:
        logger.info(f"Running pipeline for missing datasets: {missing_datasets}")

        # Step A: Augmentation
        manager.run_augmentation(
            datasets=missing_datasets,
            strategies=strategies,
            proportion=p,
            n_aug_trajs=n,
            points_proportion=pp,
            run_suffix=run_suffix,
        )

        # Step B: Feature Extraction on new data
        manager.run_aug_feature_extraction(datasets=missing_datasets, strategies=strategies, run_suffix=run_suffix)

        # Step C: Model Training & Evaluation
        new_results_path = manager.run_evaluation(
            datasets=missing_datasets, strategies=strategies, run_suffix=run_suffix
        )

        if new_results_path:
            try:
                cached_dfs.append(pd.read_csv(new_results_path))
            except Exception as e:
                logger.error(f"Failed to load newly generated results: {e}")

    # 5. Combine and Calculate Score
    if not cached_dfs:
        logger.warning("No results obtained for this trial. Returning 0.0.")
        return 0.0

    final_df = pd.concat(cached_dfs, ignore_index=True)

    os.makedirs(os.path.dirname(combined_csv_path), exist_ok=True)
    final_df.to_csv(combined_csv_path, index=False)

    aug_res = final_df[final_df["feature_type"] != "trajectory_features"]

    if aug_res.empty:
        return 0.0

    return float(aug_res["score"].mean())


def run_tuning(manager: PipelineManager, datasets: list[str], strategies: list[str]) -> None:
    """Sets up and runs the Optuna study with persistence support.

    Uses a local SQLite database to allow the study to be resumed if interrupted.
    """
    logger.info("Starting Hyperparameter Tuning (Optuna)...")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Strategies: {strategies}")

    # Define the search space for the GridSampler
    # Explicit typing to satisfy Optuna Mapping requirement
    search_space: dict[str, Sequence[str | float | int | bool | None]] = {
        "proportion": [0.1, 0.2, 0.4],
        "n_aug_trajs": [1, 3, 5],
        "points_proportion": [0.1, 0.2, 0.4],
    }

    # 1. Create a dynamic prefix (e.g., "car_traffic" or "ais_subset")
    dataset_prefix = "-".join(sorted(datasets))

    # 2. Make the Study Name unique
    study_name = f"opt_study_{dataset_prefix}"

    # 3. Make the Filename unique (Best Practice)
    db_filename = f"optuna_{dataset_prefix}.db"
    storage_path = f"sqlite:///{os.path.join(manager.output_root, db_filename)}"

    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )

    # Wrap the objective to pass our specific arguments
    study.optimize(lambda trial: objective(trial, manager, datasets, strategies))

    logger.info("=" * 30)
    logger.info("Tuning Finished Successfully")
    logger.info(f"Best Trial ID: {study.best_trial.number}")
    logger.info(f"Best F1 Score: {study.best_value:.4f}")
    logger.info("Best Parameters:")
    for key, value in study.best_params.items():
        logger.info(f"  - {key}: {value}")
    logger.info("=" * 30)
