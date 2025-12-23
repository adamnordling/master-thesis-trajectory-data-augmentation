import os
import logging
import pandas as pd
import optuna
from typing import List, Dict, Any, Optional

# We type hint using string forward reference or 'Any' to avoid circular imports at runtime
# if manager imports tuning.
from src.pipeline.manager import PipelineManager

logger = logging.getLogger(__name__)


def objective(
        trial: optuna.Trial,
        manager: PipelineManager,
        datasets: List[str],
        strategies: List[str]
) -> float:
    """
    The Optuna Objective Function.

    1. Suggests hyperparameters (Proportion, N_Augmentations, Points_Proportion).
    2. Checks if this combination already exists in the results folder (Caching).
    3. If missing, instructs the Manager to run the specific pipeline steps.
    4. Returns the average F1 score across all datasets for this configuration.
    """

    # 1. Define Search Space
    # We use categorical to allow GridSampler to work effectively
    p = trial.suggest_categorical("proportion", [0.1, 0.2, 0.4])
    n = trial.suggest_categorical("n_aug_trajs", [1, 3, 5])
    pp = trial.suggest_categorical("points_proportion", [0.1, 0.2, 0.4])

    # Create the unique suffix for filenames (e.g., _p20_n3_pp20)
    run_suffix = f"_p{int(p * 100)}_n{n}_pp{int(pp * 100)}"

    logger.info(f"--- Optuna Trial #{trial.number}: Params{run_suffix} ---")

    # 2. Check Global Cache (Combined Results)
    # If the master file for this specific config exists, we don't need to run anything.
    dataset_prefix = "-".join(sorted(datasets))
    combined_csv_path = os.path.join(
        manager.dirs["opt_history"],  # Uses the dictionary we created in Manager
        f'{dataset_prefix}{run_suffix}.csv'
    )

    if os.path.exists(combined_csv_path):
        logger.info(f"Found cached combined results for {run_suffix}. Skipping execution.")
        try:
            results_df = pd.read_csv(combined_csv_path)
            # We optimize based on the AUGMENTED performance (excluding baseline)
            aug_res = results_df[results_df['feature_type'] != 'trajectory_features']
            if not aug_res.empty:
                return aug_res['score'].mean()
        except Exception as e:
            logger.warning(f"Cache file corrupt or unreadable: {e}. Re-running.")

    # 3. Granular Check: Which datasets are missing?
    # Maybe we ran 'fox' but not 'geolife' for this config.
    missing_datasets = []
    cached_dfs = []

    for ds in datasets:
        ds_csv = os.path.join(
            manager.dirs["opt_details"], ds,
            f'{ds}{run_suffix}.csv'
        )
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
            run_suffix=run_suffix
        )

        # Step B: Feature Extraction on new data
        manager.run_aug_feature_extraction(
            datasets=missing_datasets,
            strategies=strategies,
            run_suffix=run_suffix
        )

        # Step C: Model Training & Evaluation
        new_results_path = manager.run_evaluation(
            datasets=missing_datasets,
            strategies=strategies,
            run_suffix=run_suffix
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

    # Consolidate all dataset results into one master file for this parameter set
    final_df = pd.concat(cached_dfs, ignore_index=True)

    os.makedirs(os.path.dirname(combined_csv_path), exist_ok=True)
    final_df.to_csv(combined_csv_path, index=False)

    # Calculate Optimization Metric (Mean F1 Score of Augmented Strategies)
    aug_res = final_df[final_df['feature_type'] != 'trajectory_features']

    if aug_res.empty:
        return 0.0

    return aug_res['score'].mean()


def run_tuning(manager: PipelineManager, datasets: List[str], strategies: List[str]):
    """
    Sets up and runs the Optuna study.
    """
    logger.info("Starting Hyperparameter Tuning (Optuna)...")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Strategies: {strategies}")

    # Define the search space for the GridSampler
    # This ensures we try every combination exactly once
    search_space = {
        "proportion": [0.1, 0.2, 0.4],
        "n_aug_trajs": [1, 3, 5],
        "points_proportion": [0.1, 0.2, 0.4],
    }

    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Wrap the objective to pass our specific arguments
    study.optimize(
        lambda trial: objective(trial, manager, datasets, strategies)
    )

    logger.info("=" * 30)
    logger.info("Tuning Finished Successfully")
    logger.info(f"Best Trial ID: {study.best_trial.number}")
    logger.info(f"Best F1 Score: {study.best_value:.4f}")
    logger.info("Best Parameters:")
    for key, value in study.best_params.items():
        logger.info(f"  - {key}: {value}")
    logger.info("=" * 30)