import argparse
import glob
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from scipy.stats import ttest_rel

# Config
ALPHA = 0.05
MIN_SEEDS_FOR_TEST = 5
MODEL_ORDER = ["LogisticRegression", "MLP", "RandomForest", "XGBoost"]
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import colorama

    colorama.init(autoreset=True)
    GREEN, RED, YELLOW, RESET = (
        colorama.Fore.GREEN,
        colorama.Fore.RED,
        colorama.Fore.YELLOW,
        colorama.Style.RESET_ALL,
    )
except ImportError:
    print("Colorama not found. For colored output, run: pip install colorama")
    GREEN = RED = YELLOW = RESET = ""


def find_best_configs(dataset_name: str, history_dir: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Scans all Optuna trials to find the champion hyperparameter set for each strategy."""
    search_pattern = os.path.join(history_dir, f"{dataset_name}_p*.csv")
    all_files = glob.glob(search_pattern)
    if not all_files:
        print(f"{RED}Error: No Optuna trial files found for '{dataset_name}'.{RESET}")
        return None

    strategy_champions: Dict[str, Dict[str, Any]] = {}
    param_regex = re.compile(r"(_p\d+_n\d+_pp\d+)\.csv$")

    for f in all_files:
        match = param_regex.search(f)
        if not match:
            continue
        run_suffix = match.group(1)

        df = pd.read_csv(f)

        # Vectorized string parsing for performance
        is_merged = df["feature_type"].str.contains("merged", na=False)
        df["strategy"] = "baseline"
        df.loc[is_merged, "strategy"] = df["feature_type"].str.split("merged_").str[-1].str.split("_p").str[0]

        for strategy, group in df.groupby("strategy"):
            if strategy == "baseline":
                continue

            mean_score = float(group["score"].mean())
            if strategy not in strategy_champions or mean_score > strategy_champions[strategy]["mean_score"]:
                strategy_champions[strategy] = {"mean_score": mean_score, "best_suffix": run_suffix}

    if not strategy_champions:
        print(f"{RED}Error: No valid strategies found across all Optuna trial files.{RESET}")
        return None

    return strategy_champions


def load_final_comparison_data(dataset_name: str, history_dir: str, best_params: Dict[str, Any]) -> pd.DataFrame:
    """Loads the baseline data and the best data for each strategy into one DataFrame."""
    baseline_file = os.path.join(history_dir, f"{dataset_name}_baseline_tuning.csv")
    if not os.path.exists(baseline_file):
        raise FileNotFoundError(f"FATAL: Baseline result file not found: {baseline_file}")

    baseline_df = pd.read_csv(baseline_file)
    baseline_df["strategy"] = "baseline"

    all_dfs = [baseline_df]

    for strategy, info in best_params.items():
        suffix = info["best_suffix"]
        strategy_file = os.path.join(history_dir, f"{dataset_name}{suffix}.csv")
        strategy_df = pd.read_csv(strategy_file)

        # Vectorized strategy extraction
        is_merged = strategy_df["feature_type"].str.contains("merged", na=False)
        strategy_df["strategy"] = "baseline"
        strategy_df.loc[is_merged, "strategy"] = (
            strategy_df["feature_type"].str.split("merged_").str[-1].str.split("_p").str[0]
        )

        all_dfs.append(strategy_df[strategy_df["strategy"] == strategy])

    return pd.concat(all_dfs, ignore_index=True)


def perform_statistical_analysis(df: pd.DataFrame, best_params: Dict[str, Any]) -> pd.DataFrame:
    """Performs paired t-test and returns a results DataFrame."""
    analysis_results: List[Dict[str, Any]] = []

    for model in MODEL_ORDER:
        model_df = df[df["model"] == model]

        # Use .to_numpy() to ensure we have numeric arrays with .mean() and t-test support
        baseline_scores = model_df[model_df["strategy"] == "baseline"]["score"].to_numpy(dtype=float)
        num_seeds = len(baseline_scores)

        if num_seeds == 0:
            continue

        for strategy in sorted(best_params.keys()):
            strategy_scores = model_df[model_df["strategy"] == strategy]["score"].to_numpy(dtype=float)

            if len(baseline_scores) != len(strategy_scores):
                continue

            mean_improvement = strategy_scores.mean() - baseline_scores.mean()

            if num_seeds < MIN_SEEDS_FOR_TEST:
                p_value, is_significant, status = (float("nan"), False, f"Too few seeds ({num_seeds})")
            else:
                # Scipy ttest_rel is safe here as we ensured numpy arrays
                _, p_value = ttest_rel(strategy_scores, baseline_scores)
                is_significant = bool(p_value < ALPHA)
                status = "OK"

            analysis_results.append({
                "model": model,
                "strategy": strategy,
                "mean_improvement_pct": mean_improvement * 100,
                "p_value": p_value,
                "is_significant": is_significant,
                "significant_improvement": is_significant and mean_improvement > 0,
                "num_seeds": num_seeds,
                "status": status,
                "best_suffix": best_params[strategy]["best_suffix"],
            })

    return pd.DataFrame(analysis_results)


def save_artifacts(results_df: pd.DataFrame, dataset_name: str, best_params: Dict[str, Any]) -> None:
    """
    Saves results to console, CSV, LaTeX, and Markdown files.

    Generates a professional reports directory structure and populates it with analysis files.
    """
    output_dir = os.path.join(PROJECT_ROOT, "data", "output", "analysis", "reports", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    latex_path = os.path.join(output_dir, f"{dataset_name}_final_table.tex")

    caption_text = (
        "\\caption{\n"
        "        Statistical analysis of the performance improvement for each optimally-tuned \n"
        "        augmentation strategy compared to the baseline. This table presents the final, \n"
        "        rigorous comparison, in contrast to the exploratory overview in the previous table.\n"
        "        An asterisk (*) denotes a statistically significant result (p < 0.05).\n"
        "    }"
    )

    with open(latex_path, "w", encoding="utf-8") as f:
        max_seeds = int(results_df["num_seeds"].max()) if not results_df.empty else 0
        f.write(f"% Auto-generated for {dataset_name} with {max_seeds} seeds.\n")
        f.write("\\begin{table}[ht]\n")
        f.write("    \\centering\n")
        f.write(f"    {caption_text}\n")
        f.write("    \\label{tab:final_analysis_" + dataset_name.lower() + "}\n")

        col_spec = "{l l r r}"

        f.write(f"    \\begin{{tabular}}{{{col_spec}}}\n")
        f.write("        \\toprule\n")
        f.write(
            "        \\textbf{Model} & \\textbf{Strategy} & \\textbf{Mean Improv. (\\%)} & \\textbf{p-value} \\\\\n"
        )
        f.write("        \\midrule\n")

        for i, model in enumerate(MODEL_ORDER):
            model_results = results_df[results_df["model"] == model]
            if model_results.empty:
                continue
            if i > 0:
                f.write("        \\midrule\n")

            f.write(f"        \\multirow{{{len(model_results)}}}{{*}}{{\\textbf{{{model}}}}} \n")

            for _, row in model_results.iterrows():
                strategy_name = str(row["strategy"]).replace("_", " ").title()
                imp_val_str = f"{row['mean_improvement_pct']:+.2f}"

                if pd.isna(row["p_value"]):
                    p_val_str = "---"
                else:
                    p_val_str = f"{row['p_value']:.4f}"
                    if row["p_value"] < 0.0001:
                        p_val_str = "<0.0001"
                    if row["significant_improvement"]:
                        p_val_str += "*"

                if row["significant_improvement"]:
                    f.write(
                        f"        & \\textbf{{{strategy_name}}} & \\textbf{{{imp_val_str}}} & \\textbf{{{p_val_str}}} \\\\\n"
                    )
                else:
                    f.write(f"        & {strategy_name} & {imp_val_str} & {p_val_str} \\\\\n")

        f.write("        \\bottomrule\n")
        f.write("    \\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"{GREEN}Saved SIMPLIFIED & ROBUST LaTeX table to:{RESET} {latex_path}")


def main(dataset_name: str) -> None:
    """Main execution function, designed to be called from other scripts."""
    history_dir = os.path.join(PROJECT_ROOT, "data", "output", "optimization", "history")

    best_params = find_best_configs(dataset_name, history_dir)
    if best_params:
        try:
            final_df = load_final_comparison_data(dataset_name, history_dir, best_params)
            analysis_df = perform_statistical_analysis(final_df, best_params)
            if not analysis_df.empty:
                save_artifacts(analysis_df, dataset_name, best_params)
            else:
                print(
                    f"{YELLOW}Analysis completed, but no comparable results were found. Check for seed mismatches.{RESET}"
                )
        except FileNotFoundError as e:
            print(f"{RED}{e}{RESET}")
        except Exception as e:
            print(f"{RED}An unexpected error occurred during analysis: {e}{RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the final analysis pipeline for the thesis.")
    parser.add_argument("--dataset", type=str, required=True, help='Name of the dataset (e.g., "Foxes").')
    args = parser.parse_args()
    main(args.dataset)
