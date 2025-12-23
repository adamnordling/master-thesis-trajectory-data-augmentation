import os
import gc
import logging
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Dict, Any, Tuple

# --- Internal Imports ---
from src.core.features import extract_trajectory_features
from src.generators.augmentor import apply_augmentation_pipeline
from src.strategies.factory import get_strategy_class
from src.utils import load_dataframe, save_dataframe

# Initialize Logger
logger = logging.getLogger(__name__)

# --- PATH CONFIGURATION ---
# Determine project root relative to this file: src/pipeline/workers.py -> ../../
current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Define standard data directories
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
AUG_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'augmented')


def prepare_dataset_worker(task_args: Tuple) -> Dict[str, Any]:
    """
    Worker: Reads raw CSV (data/raw), cleans it, splits into Train/Test,
    and saves as .feather files to data/augmented/{dataset}/{seed}/.

    Args:
        task_args: Tuple containing (seed, dataset_name, dataset_config_dict)
    """
    seed, dataset_name, dataset_config = task_args

    try:
        # Output directory: data/augmented/{dataset}/{seed}/
        output_dir = os.path.join(AUG_DATA_DIR, dataset_name, str(seed))

        # Input path: data/raw/{filename}
        # Note: Raw data is always CSV
        raw_filepath = os.path.join(RAW_DATA_DIR, dataset_config['csv_file'])

        # 1. Load Raw Data
        # We handle raw loading manually here because raw CSVs often lack headers
        # or require specific dtype mapping that generic load_dataframe might miss.
        try:
            df = pd.read_csv(raw_filepath, dtype={'tid': str})
        except ValueError:
            # Fallback for datasets without headers (assume col 0 is TID)
            logger.info(f"[{dataset_name}] No header found. Using col 0 as TID.")
            df = pd.read_csv(raw_filepath, dtype={0: str})
            df.rename(columns={df.columns[0]: 'tid'}, inplace=True)

        # 2. Cleaning
        initial_rows = len(df)
        df.dropna(subset=['lat', 'lon', 'label'], inplace=True)
        if len(df) < initial_rows:
            logger.debug(f"[{dataset_name}] Dropped {initial_rows - len(df)} rows with missing values.")

        df = df.sort_values(by=['tid', 'time'])

        # Identify trajectories that never actually move (Lat and Lon never change)
        traj_variance = df.groupby('tid')[['lat', 'lon']].nunique()
        stationary_tids = traj_variance[(traj_variance['lat'] <= 1) & (traj_variance['lon'] <= 1)].index

        if not stationary_tids.empty:
            logger.info(f"[{dataset_name}] Removing {len(stationary_tids)} stationary trajectories (no movement).")
            df = df[~df['tid'].isin(stationary_tids)]
        # ---------------------------

        # 3. Filter single-trajectory classes
        # StratifiedShuffleSplit throws errors if a class has only 1 member.
        traj_labels = df.drop_duplicates(subset='tid').set_index('tid')['label']
        class_counts = traj_labels.value_counts()
        single_member_classes = class_counts[class_counts < 2].index.tolist()

        if single_member_classes:
            logger.info(f"[{dataset_name}] Exlcuding {len(single_member_classes)} classes with < 2 trajectories.")
            tids_to_exclude = traj_labels[traj_labels.isin(single_member_classes)].index
            df = df[~df['tid'].isin(tids_to_exclude)]
            traj_labels = traj_labels.drop(tids_to_exclude)

        if traj_labels.empty:
            raise ValueError(f"No valid trajectories remaining for {dataset_name} after cleaning.")

        # 4. Stratified Split
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(splitter.split(traj_labels.index, traj_labels.values))

        train_tids = traj_labels.index[train_idx]
        test_tids = traj_labels.index[test_idx]

        train_df = df[df['tid'].isin(train_tids)].copy()
        test_df = df[df['tid'].isin(test_tids)].copy()
        del df  # Clear the large raw dataset from memory

        # 5. Save (Feather Only)
        # Using save_dataframe utility ensures directory creation and clean saves
        save_dataframe(train_df, os.path.join(output_dir, f"train_seed_{seed}_pts.feather"))
        save_dataframe(test_df, os.path.join(output_dir, f"test_seed_{seed}_pts.feather"))

        return {"success": True, "task": f"Prepare {dataset_name} seed {seed}"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "task": f"Prepare {dataset_name}", "error": str(e)}
    finally:
        gc.collect()

def extract_features_worker(filepath: str) -> Dict[str, Any]:
    """
    Worker: Loads a trajectory points file, calculates features, and saves them.

    Args:
        filepath: Full path to the _pts.feather file.
    """
    try:
        # 1. Load using robust utility (Handles feather automatically)
        df = load_dataframe(filepath)

        # 2. Run Core Feature Extraction logic
        features_df = extract_trajectory_features(df)

        # 3. Generate Output Path
        # e.g., train_seed_1415_pts.feather -> train_seed_1415_pts_trajectory_features.feather
        base_path = os.path.splitext(filepath)[0]
        output_base = f"{base_path}_trajectory_features.feather"

        # 4. Save
        save_dataframe(features_df, output_base)

        return {"success": True, "task": os.path.basename(filepath)}

    except Exception as e:
        return {"success": False, "task": filepath, "error": str(e)}
    finally:
        gc.collect()

def augment_data_worker(task_args: Tuple) -> Dict[str, Any]:
    """
    Worker: Executes the Augmentation Pipeline.

    1. Loads Train Points and Train Features (Feather).
    2. Instantiates the selection Strategy.
    3. Runs the Augmentation Generator.
    4. Saves the new augmented dataset (Feather).

    Args:
        task_args: (seed, strategy_name, proportion, n_aug, points_prop, dataset_name, run_suffix)
    """
    seed, strategy_name, proportion, n_aug_trajs, points_proportion, dataset_name, run_suffix = task_args

    try:
        folder = os.path.join(AUG_DATA_DIR, dataset_name, str(seed))

        # 1. Resolve Paths
        pts_path = os.path.join(folder, f"train_seed_{seed}_pts.feather")
        feats_path = os.path.join(folder, f"train_seed_{seed}_pts_trajectory_features.feather")

        # 2. Load Data
        pts_df = load_dataframe(pts_path)
        feats_df = load_dataframe(feats_path)

        # 3. Instantiate Strategy via Factory
        # Note: We pass the main 'proportion' here.
        # Detailed params (like k-clusters) are handled by defaults in the class.
        StrategyClass = get_strategy_class(strategy_name)
        strategy_instance = StrategyClass(proportion=proportion, seed=seed)

        # 4. Run Generator Logic
        # This returns a DF with augmented=0 (original) and augmented=1..N (new)
        augmented_df = apply_augmentation_pipeline(
            trajectory_points_df=pts_df,
            trajectory_feats_df=feats_df,
            strategy=strategy_instance,
            n_aug_trajs=n_aug_trajs,
            points_proportion=points_proportion,
            seed=seed
        )

        # 5. Save Output
        output_path = os.path.join(folder, f"train_seed_{seed}_pts_augmented_{strategy_name}{run_suffix}.feather")
        save_dataframe(augmented_df, output_path)

        return {"success": True, "task": f"Augment {dataset_name} {strategy_name}"}

    except Exception as e:
        return {"success": False, "task": f"Augment {dataset_name}", "error": str(e)}
    finally:
        gc.collect()

def extract_aug_features_worker(task_args: Tuple) -> Dict[str, Any]:
    """
    Worker: Optimized feature extraction for augmented data.

    Instead of recalculating features for the whole dataset, it:
    1. Loads the augmented dataset.
    2. Identifies ONLY the newly generated trajectories (augmented != 0).
    3. Calculates features for them.
    4. Merges them with the existing features of the original data.
    """
    seed, strategy, dataset_name, run_suffix = task_args
    try:
        folder = os.path.join(AUG_DATA_DIR, dataset_name, str(seed))

        # Paths
        aug_pts_path = os.path.join(folder, f'train_seed_{seed}_pts_augmented_{strategy}{run_suffix}.feather')
        orig_feats_path = os.path.join(folder, f'train_seed_{seed}_pts_trajectory_features.feather')

        # Load
        aug_df = load_dataframe(aug_pts_path)
        orig_feats_df = load_dataframe(orig_feats_path)

        # 1. Filter for New Data (augmented != 0)
        aug_only_df = aug_df[aug_df['augmented'] != 0].copy()

        if aug_only_df.empty:
            # If no augmentation happened (e.g. strategy selected 0), just return original
            combined_df = orig_feats_df.copy()
        else:
            # 2. Ensure Unique TIDs
            # We append the augmentation ID to the TID to make it unique (e.g., "123_1")
            # This is critical for the feature extraction to group correctly
            # --- NEW CODE (Bulletproof string concatenation) ---
            # Using .str.cat ensures that if TID was "001", it stays "001_1"
            # and never becomes "1_1" or "1.0_1"
            aug_only_df['tid'] = aug_only_df['tid'].astype(str).str.cat(
                aug_only_df['augmented'].astype(str), sep='_'
            )

            # Clean up columns not needed for extraction
            aug_only_df = aug_only_df.drop(columns=['augmented'])

            # 3. Extract Features on Subset
            new_features_df = extract_trajectory_features(aug_only_df)

            # 4. Merge
            # Ensure TIDs are strings for clean concatenation
            new_features_df['tid'] = new_features_df['tid'].astype(str)
            orig_feats_df['tid'] = orig_feats_df['tid'].astype(str)

            combined_df = pd.concat([orig_feats_df, new_features_df], ignore_index=True)

        # 5. Save Merged Result
        output_path = os.path.join(folder,
                                   f'train_seed_{seed}_pts_trajectory_features_merged_{strategy}{run_suffix}.feather')
        save_dataframe(combined_df, output_path)

        return {"success": True, "task": f"ExtractAug {dataset_name} {strategy}"}

    except Exception as e:
        return {"success": False, "task": f"ExtractAug {dataset_name}", "error": str(e)}
    finally:
        gc.collect()