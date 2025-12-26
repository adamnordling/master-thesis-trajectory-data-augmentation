import argparse
import logging
import os
import sys
from typing import List

from src.utils.profiler import PerformanceTracker

# Ensure src is in python path for module discovery
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.manager import PipelineManager
from src.pipeline.tuning import run_tuning
from src.utils.config import load_config
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Defines and parses the command-line arguments for the pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Professional Trajectory Data Augmentation Pipeline"
    )

    # --- Workflow Flags (Step-by-Step execution) ---
    parser.add_argument(
        "--prepare", action="store_true", help="Step 1: Prepare datasets (Split/Clean)"
    )
    parser.add_argument(
        "--extract", action="store_true", help="Step 2: Extract features from raw data"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Step 3: Generate augmented data (Manual mode)",
    )
    parser.add_argument(
        "--extract-augmented",
        action="store_true",
        help="Step 4: Extract features from augmented data",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Step 5: Run evaluation manually"
    )
    parser.add_argument(
        "--report", action="store_true", help="Step 6: Generate final reports"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Step 7: Run the complete final analysis and generate all reports",
    )

    # --- Filters ---
    parser.add_argument(
        "--datasets",
        type=str,
        help='Comma-separated list of datasets to process (e.g., "fox,geolife")',
    )
    parser.add_argument(
        "--strategies",
        type=str,
        help='Comma-separated list of strategies (e.g., "random,diversity")',
    )

    # --- Manual Augmentation Parameters ---
    parser.add_argument(
        "--prop",
        type=float,
        default=0.2,
        help="Proportion of trajectories to select (default: 0.2)",
    )
    parser.add_argument(
        "--n-aug",
        type=int,
        default=3,
        help="Number of augmentations per trajectory (default: 3)",
    )
    parser.add_argument(
        "--points-prop",
        type=float,
        default=0.2,
        help="Proportion of points to shift (default: 0.2)",
    )

    # --- System & Debugging ---
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode (uses first 2 seeds only)"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_false",
        dest="parallel",
        help="Disable parallel processing (for debugging)",
    )

    return parser.parse_args()


def main() -> None:
    """
    Orchestrates the Trajectory Augmentation Pipeline.
    """
    profiler = PerformanceTracker()
    setup_logging()
    args = parse_args()

    logger.info("Initializing Trajectory Augmentation Pipeline...")

    try:
        config = load_config("config")
    except Exception as e:
        logger.critical(f"Failed to load configuration: {e}")
        sys.exit(1)

    if args.test:
        logger.warning("--- RUNNING IN TEST MODE (First 2 Seeds Only) ---")
        config["seeds"] = config["seeds"][:2]
    if not args.parallel:
        logger.warning("Parallel processing disabled by user.")
        config["processing"]["parallel"] = False

    manager = PipelineManager(config)

    if args.datasets:
        datasets_to_process = [d.strip() for d in args.datasets.split(",")]
    else:
        datasets_to_process = list(manager._discover_raw_datasets().keys())

    if args.strategies:
        strategies_to_process = [s.strip() for s in args.strategies.split(",")]
    else:
        strategies_to_process = config["strategy_config"]["active_strategies"]

    if not datasets_to_process:
        logger.error("No valid datasets found to process. Check data/raw/.")
        sys.exit(1)

    profiler.set_info("target_datasets", datasets_to_process)
    profiler.set_info(
        "parallel_workers_used",
        manager._get_max_workers() if config["processing"]["parallel"] else 1,
    )

    try:
        logger.info(f"Target Datasets: {datasets_to_process}")
        logger.info(f"Target Strategies: {strategies_to_process}")

        # --- SPECIAL CASE: Standalone Analysis ---
        if args.analyze:
            logger.info("--- Running Final Analysis Only ---")
            manager.run_final_analysis(datasets_to_process)
            sys.exit(0)

        manual_steps_requested = any(
            [args.prepare, args.extract, args.augment, args.extract_augmented, args.evaluate, args.report]
        )

        if manual_steps_requested:
            # --- MANUAL MODE (Dataset Isolation Loop) ---
            logger.info("--- Running in Manual Mode ---")
            run_suffix = f"_p{int(args.prop * 100)}_n{args.n_aug}_pp{int(args.points_prop * 100)}"

            if args.prepare:
                manager.run_preparation(datasets_to_process)
            if args.extract:
                manager.run_feature_extraction(datasets_to_process)

            # Sub-steps are iterated to prevent file collisions
            for ds in datasets_to_process:
                ds_list = [ds]
                if args.augment:
                    manager.run_augmentation(
                        ds_list, strategies_to_process, args.prop, args.n_aug, args.points_prop, run_suffix
                    )
                if args.extract_augmented:
                    manager.run_aug_feature_extraction(ds_list, strategies_to_process, run_suffix)
                if args.evaluate:
                    manager.run_evaluation(ds_list, strategies_to_process, run_suffix)
                if args.report:
                    manager.generate_reports(ds_list)
            if args.report:
                logger.info("Generating combined LaTeX table for all datasets...")
                manager.generate_reports(datasets_to_process)

        else:
            # --- AUTOMATIC MODE (The Specialist Loop) ---
            logger.info("--- Running in Automatic Mode (Full Optuna Loop with Isolation) ---")

            # Bulk processing for Step 1 & 2 (maximize CPU usage)
            profiler.start("1_Bulk_Preparation")
            manager.run_preparation(datasets_to_process)
            profiler.stop("1_Bulk_Preparation")

            profiler.start("2_Bulk_Feature_Extraction")
            manager.run_feature_extraction(datasets_to_process)
            profiler.stop("2_Bulk_Feature_Extraction")

            # Sequential Loop for Tuning/Analysis to ensure specialist parameters
            for ds in datasets_to_process:
                logger.info(f"\n{'=' * 60}\nSTARTING MISSION: {ds.upper()}\n{'=' * 60}")
                ds_single = [ds]

                # Step 3: Specialist Baseline Tuning
                profiler.start(f"3_Baseline_Tuning_{ds}")
                params_path = os.path.join(manager.dirs["states"], f"{ds}_best_params.csv")
                results_path = os.path.join(manager.dirs["opt_history"], f"{ds}_baseline_tuning.csv")

                if not (os.path.exists(params_path) and os.path.exists(results_path)):
                    logger.info(f"Baseline missing or incomplete for {ds}. Tuning...")
                    manager.run_evaluation(ds_single, strategies=[], run_suffix="_baseline_tuning")
                else:
                    logger.info(f"Baseline for {ds} already exists. Skipping Step 3.")
                profiler.stop(f"3_Baseline_Tuning_{ds}")

                # Step 4: Specialist Optuna Tuning (Unique DB for this dataset)
                profiler.start(f"4_Optuna_Loop_{ds}")
                run_tuning(manager, ds_single, strategies_to_process)
                profiler.stop(f"4_Optuna_Loop_{ds}")

                # Step 5: Specialist Reporting & Statistical Analysis
                logger.info(f"Generating final reports for {ds}...")
                profiler.start(f"5_Reporting_{ds}")
                manager.generate_reports(ds_single)
                manager.run_final_analysis(ds_single)
                profiler.stop(f"5_Reporting_{ds}")

                logger.info(f"FINISHED ALL STEPS FOR: {ds}")
        logger.info("\nMISSION COMPLETE: Generating combined final summary table...")
        manager.generate_reports(datasets_to_process)

    finally:
        if datasets_to_process:
            profiler.generate_report(datasets_to_process)


if __name__ == "__main__":
    main()