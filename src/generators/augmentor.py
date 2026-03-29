import logging
import traceback

import pandas as pd

# Import our modular components
from src.core.geometry import shift_points_randomly
from src.strategies.base import TrajectorySelectionMethod

# Initialize logger
logger = logging.getLogger(__name__)


def apply_augmentation_pipeline(
    trajectory_points_df: pd.DataFrame,
    trajectory_feats_df: pd.DataFrame,
    strategy: TrajectorySelectionMethod,
    n_aug_trajs: int,
    points_proportion: float,
    seed: int,
) -> pd.DataFrame:
    """Coordinator function that runs the full augmentation pipeline on a dataset.

    1. Uses the Strategy to select TIDs from the features DataFrame.
    2. Filters the raw points DataFrame to those TIDs.
    3. Calls the Geometry core to generate shifted variations.
    4. Merges augmented data with the untouched original data.

    Args:
        trajectory_points_df: Raw trajectory data (lat, lon, time, tid).
        trajectory_feats_df: Calculated features (speed, angle, etc.).
        strategy: An instance of a TrajectorySelectionMethod subclass.
        n_aug_trajs: Number of new trajectories to generate per selected ID.
        points_proportion: Fraction of points to shift.
        seed: Random seed.

    Returns:
        A single DataFrame containing both the original and augmented data.
    """
    logger.info(f"Starting augmentation pipeline with strategy: {strategy.__class__.__name__}")
    logger.debug(f"Input points shape: {trajectory_points_df.shape}")
    logger.debug(f"Input features shape: {trajectory_feats_df.shape}")

    # Step 1: Selection
    # The strategy analyzes features and returns a list of TIDs
    trajs_tids_to_augment = strategy.select(trajectory_feats_df)
    logger.info(f"Strategy selected {len(trajs_tids_to_augment)} trajectories for augmentation.")

    # Filter raw points to just the selected trajectories
    selected_trajs_points_df = trajectory_points_df[trajectory_points_df["tid"].isin(trajs_tids_to_augment)].copy()

    # Step 2: Edge Case Handling
    # If the strategy returned nothing (e.g., extremely strict filtering), return original data
    if selected_trajs_points_df.empty:
        logger.warning("No trajectories selected for augmentation. Returning original dataset unchanged.")
        original_df = trajectory_points_df.copy()
        if "augmented" not in original_df.columns:
            original_df["augmented"] = 0
        return original_df

    # Step 3: Augmentation
    try:
        # shift_points_randomly returns a DF with:
        # 1. The original selected trajectories (augmented=0)
        # 2. The new shifted versions (augmented=1..N)
        augmented_df = shift_points_randomly(
            trajectory_df=selected_trajs_points_df,
            n_aug_trajs=n_aug_trajs,
            points_proportion=points_proportion,
            seed=seed,
        )
        logger.debug(f"Augmentation core produced {len(augmented_df)} rows.")

    except Exception as e:
        logger.error(f"Critical error in augmentation core: {e}")
        logger.debug(traceback.format_exc())

        # Fallback: Return original data to prevent pipeline crash
        original_df = trajectory_points_df.copy()
        original_df["augmented"] = 0
        return original_df

    # Step 4: Re-assembly
    # We have the augmented selected data. Now we need the unselected data.
    non_selected_tids = [tid for tid in trajectory_points_df["tid"].unique() if tid not in trajs_tids_to_augment]

    non_selected_df = trajectory_points_df[trajectory_points_df["tid"].isin(non_selected_tids)].copy()

    # Ensure consistency: unselected data is "original", so augmented=0
    non_selected_df["augmented"] = 0

    # Concatenate everything
    final_df = pd.concat([augmented_df, non_selected_df], ignore_index=True)

    logger.info(f"Pipeline complete. Final dataset shape: {final_df.shape}")
    before = trajectory_points_df.drop_duplicates("tid")["label"].value_counts(normalize=True)
    after = final_df.drop_duplicates("tid")["label"].value_counts(normalize=True)
    logger.info(f"Class distribution check (Before vs After): \n{before} \nvs\n {after}")

    return final_df
