import argparse
import logging
import sys
import os
from src.utils.profiler import PerformanceTracker # Import the new class

# --- OPTIMIZATION: Intel Hardware Acceleration (Silent Mode) ---
# We set this variable to turn off the "Intel(R) Extension..." spam
logging.getLogger("sklearnex").setLevel(logging.ERROR)
try:
    from sklearnex import patch_sklearn
    # verbose=False suppresses the startup message
    patch_sklearn()
except ImportError:
    pass
# ---------------------------------------------------------------

# Ensure src is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.pipeline.manager import PipelineManager
from src.pipeline.tuning import run_tuning

logger = logging.getLogger(__name__)


def parse_args():
    """
    Defines the command-line arguments for the pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Professional Trajectory Data Augmentation Pipeline'
    )

    # --- Workflow Flags (Step-by-Step execution) ---
    parser.add_argument('--prepare', action='store_true', help='Step 1: Prepare datasets (Split/Clean)')
    parser.add_argument('--extract', action='store_true', help='Step 2: Extract features from raw data')
    parser.add_argument('--augment', action='store_true', help='Step 3: Generate augmented data (Manual mode)')
    parser.add_argument('--extract-augmented', action='store_true', help='Step 4: Extract features from augmented data')
    parser.add_argument('--evaluate', action='store_true', help='Step 5: Run evaluation manually')
    parser.add_argument('--report', action='store_true', help='Step 6: Generate final reports')
    parser.add_argument('--analyze', action='store_true',
                        help='Step 7: Run the complete final analysis and generate all reports')

    # --- Filters ---
    parser.add_argument('--datasets', type=str,
                        help='Comma-separated list of datasets to process (e.g., "fox,geolife")')
    parser.add_argument('--strategies', type=str, help='Comma-separated list of strategies (e.g., "random,diversity")')

    # --- Manual Augmentation Parameters (Ignored if running Optuna) ---
    parser.add_argument('--prop', type=float, default=0.2, help='Proportion of trajectories to select (default: 0.2)')
    parser.add_argument('--n-aug', type=int, default=3, help='Number of augmentations per trajectory (default: 3)')
    parser.add_argument('--points-prop', type=float, default=0.2, help='Proportion of points to shift (default: 0.2)')

    # --- System & Debugging ---
    parser.add_argument('--test', action='store_true', help='Run in test mode (uses first 2 seeds only)')
    parser.add_argument('--no-parallel', action='store_false', dest='parallel',
                        help='Disable parallel processing (for debugging)')

    return parser.parse_args()


# The parse_args function does not need to change.
# Only the main function below is updated.

def main():
    # Initialize Profiler at the very beginning of the run
    profiler = PerformanceTracker()

    # Setup Logging and load configuration
    setup_logging()
    args = parse_args()
    logger.info("Initializing Trajectory Augmentation Pipeline...")
    try:
        config = load_config("config")
    except Exception as e:
        logger.critical(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Apply CLI Overrides to Config
    if args.test:
        logger.warning("--- RUNNING IN TEST MODE (First 2 Seeds Only) ---")
        config['seeds'] = config['seeds'][:2]
    if not args.parallel:
        logger.warning("Parallel processing disabled by user.")
        config['processing']['parallel'] = False

    # Initialize Pipeline Manager and determine scope
    manager = PipelineManager(config)
    if args.datasets:
        datasets_to_process = [d.strip() for d in args.datasets.split(',')]
    else:
        datasets_to_process = list(manager._discover_raw_datasets().keys())
    if args.strategies:
        strategies_to_process = [s.strip() for s in args.strategies.split(',')]
    else:
        strategies_to_process = config['strategy_config']['active_strategies']
    if not datasets_to_process:
        logger.error("No valid datasets found to process. Please check data/raw/ or --datasets arg.")
        sys.exit(1)

    # Log key info to the profiler
    profiler.set_info('target_datasets', datasets_to_process)
    profiler.set_info('parallel_workers_used', manager._get_max_workers() if config['processing']['parallel'] else 1)

    # Use a try...finally block to GUARANTEE the performance report is generated, even on crash
    try:
        logger.info(f"Target Datasets: {datasets_to_process}")
        logger.info(f"Target Strategies: {strategies_to_process}")

        # --- EXECUTION LOGIC ---
        # Final Analysis is a special, standalone step
        if args.analyze:
            logger.info("--- Entering Step 7: Final Analysis and Report Generation ---")
            manager.run_final_analysis(datasets_to_process)
            logger.info("Analysis complete. Exiting.")
            sys.exit(0)

        manual_steps_requested = any([
            args.prepare, args.extract, args.augment,
            args.extract_augmented, args.evaluate, args.report
        ])

        if manual_steps_requested:
            # --- MANUAL MODE with PROFILING ---
            logger.info("--- Running in Manual Mode (Executing specified steps) ---")
            run_suffix = f"_p{int(args.prop * 100)}_n{args.n_aug}_pp{int(args.points_prop * 100)}"
            if args.prepare:
                profiler.start('1_Preparation');
                manager.run_preparation(datasets_to_process);
                profiler.stop('1_Preparation')
            if args.extract:
                profiler.start('2_Base_Feature_Extraction');
                manager.run_feature_extraction(datasets_to_process);
                profiler.stop('2_Base_Feature_Extraction')
            if args.augment:
                profiler.start('3_Augmentation');
                manager.run_augmentation(datasets_to_process, strategies_to_process, proportion=args.prop,
                                         n_aug_trajs=args.n_aug, points_proportion=args.points_prop,
                                         run_suffix=run_suffix);
                profiler.stop('3_Augmentation')
            if args.extract_augmented:
                profiler.start('4_Augmented_Feature_Extraction');
                manager.run_aug_feature_extraction(datasets_to_process, strategies_to_process, run_suffix=run_suffix);
                profiler.stop('4_Augmented_Feature_Extraction')
            if args.evaluate:
                profiler.start('5_Evaluation');
                manager.run_evaluation(datasets_to_process, strategies_to_process, run_suffix=run_suffix);
                profiler.stop('5_Evaluation')
            if args.report:
                profiler.start('6_Visual_Report_Generation');
                manager.generate_reports(datasets_to_process);
                profiler.stop('6_Visual_Report_Generation')

        else:
            # --- AUTOMATIC MODE (Optuna) with PROFILING ---
            logger.info("--- Running in Automatic Mode (Full Optuna Optimization Loop) ---")

            profiler.start('1_Preparation');
            manager.run_preparation(datasets_to_process);
            profiler.stop('1_Preparation')
            profiler.start('2_Base_Feature_Extraction');
            manager.run_feature_extraction(datasets_to_process);
            profiler.stop('2_Base_Feature_Extraction')

            profiler.start('3_Baseline_Model_Tuning')
            need_baseline_tuning = any(
                not os.path.exists(os.path.join(manager.dirs["states"], f'{ds}_best_params.csv')) for ds in
                datasets_to_process)
            if need_baseline_tuning:
                logger.info("Baseline models not tuned yet. Running baseline evaluation...")
                manager.run_evaluation(datasets_to_process, strategies=[], run_suffix="_baseline_tuning")
            profiler.stop('3_Baseline_Model_Tuning')

            profiler.start('4_Optuna_Tuning_Loop')
            run_tuning(manager, datasets_to_process, strategies_to_process)
            profiler.stop('4_Optuna_Tuning_Loop')

            logger.info("--- Optuna tuning finished. Automatically generating all final reports. ---")
            profiler.start('5a_Visual_Reports');
            manager.generate_reports(datasets_to_process);
            profiler.stop('5a_Visual_Reports')
            profiler.start('5b_Statistical_Analysis');
            manager.run_final_analysis(datasets_to_process);
            profiler.stop('5b_Statistical_Analysis')

    finally:
        # This block will run regardless of whether the pipeline succeeded or failed
        if datasets_to_process:
            profiler.generate_report(datasets_to_process)


if __name__ == "__main__":
    main()
