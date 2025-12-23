import os
import re
import glob
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

# Set backend to Agg (Anti-Grain Geometry) to render plots without a GUI
# This prevents crashes on servers or when running in background processes
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from typing import List, Dict, Optional

# Initialize logger
logger = logging.getLogger(__name__)


def _clean_strategy_name(strategy: str) -> str:
    """Helper to format strategy names for plots."""
    if strategy == 'baseline':
        return 'Baseline'
    return strategy.capitalize()



def _load_and_parse_results(results_dir: str, datasets: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Scans the results directory for CSVs and robustly parses hyperparameters
    from both baseline and Optuna trial filenames.
    """
    history_dir = os.path.join(results_dir, "optimization", "history")

    # If specific datasets are given, search for them. Otherwise, get all files.
    if datasets:
        all_files = []
        for ds in datasets:
            all_files.extend(glob.glob(os.path.join(history_dir, f'{ds}*.csv')))
    else:
        all_files = glob.glob(os.path.join(history_dir, '*.csv'))

    if not all_files:
        logger.error(f"No result CSVs found in {history_dir}")
        return pd.DataFrame()

    df_list = []
    # Regex to find hyperparameter values like _p10_n3_pp20 in a filename
    param_regex = re.compile(r'_p(\d+)_n(\d+)_pp(\d+)\.csv$')

    for f in all_files:
        try:
            df = pd.read_csv(f)
            filename = os.path.basename(f)

            # Use the regex to find a match
            match = param_regex.search(filename)

            if match:
                # If it's an Optuna file (e.g., Foxes_p10_n3_pp20.csv), parse the params
                p = int(match.group(1)) / 100
                n = int(match.group(2))
                pp = int(match.group(3)) / 100
            else:
                # If it's any other file (like the baseline), set params to NaN
                p, n, pp = np.nan, np.nan, np.nan

            df['proportion'] = p
            df['n_aug_trajs'] = n
            df['points_proportion'] = pp
            df_list.append(df)

        except Exception as e:
            logger.warning(f"Could not parse file {f}: {e}")
            continue

    if not df_list:
        return pd.DataFrame()

    combined_df = pd.concat(df_list, ignore_index=True)

    # Extract clean strategy name from feature_type column
    combined_df['strategy'] = combined_df['feature_type'].apply(
        lambda ft: ft.split('merged_')[-1].split('_p')[0] if 'merged' in str(ft) else 'baseline'
    )

    return combined_df


# In src/evaluation/reporting.py

def _generate_latex_table(
        df: pd.DataFrame,
        best_params: Dict,
        output_path: str
):
    """
    Generates the LaTeX table comparing models and strategies, matching the
    exact thesis format including siunitx, resizebox, and explicit % signs.
    """
    # This guarantees the columns are always in the order you want for the table.
    strategies_list = ['baseline', 'diversity', 'outlierness', 'random', 'representativeness', 'uncertainty']

    pivot_table = df.groupby(['dataset', 'model', 'strategy'])['score'].mean().reset_index()

    datasets_list = sorted(pivot_table['dataset'].unique())
    models_list = ['LogisticRegression', 'MLP', 'RandomForest', 'XGBoost']
    model_abbreviations = {
        'LogisticRegression': 'LogisticReg',
        'RandomForest': 'RandomForest',
        'XGBoost': 'XGBoost',
        'MLP': 'MLP'
    }

    caption_text = (
        "\\caption{Performance comparison using the optimal augmentation hyperparameters identified for each dataset. "
        "Each score represents the weighted F1-score, averaged over all seeds, for that specific model and strategy. "
        "Scores higher than the baseline are in \\textbf{bold}.\\\\ \n"
        "    For each dataset, the selected parameters are shown in parentheses:\\\\ \n"
        "    \\textit{{Sel} = proportion of trajectories selected}\\\\ \n"
        "    \\textit{{Augs} = number of augmentations per trajectory}\\\\ \n"
        "    \\textit{{Pts} = proportion of points modified.}}\n"
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("    \\centering\n")
        f.write(caption_text)
        f.write("    \\label{tab:final_results}\n\n")

        f.write("    \\sisetup{table-format=2.2, table-space-text-post={\\%}}\n\n")
        f.write("    \\resizebox{\\linewidth}{!}{\n")

        col_spec = "l S S S S S S"

        f.write(f"    \\begin{{tabular}}{{{col_spec}}}\n")
        f.write("        \\toprule\n")

        header = (
            "        \\textbf{Model} & {\\textbf{Baseline}} & {\\textbf{Diversity}} & "
            "{\\textbf{Outlierness}} & {\\textbf{Random}} & "
            "{\\textbf{Representativeness}} & {\\textbf{Uncertainty}} \\\\\n"
        )
        f.write(header)
        f.write("        \\midrule\n")

        for i, dataset in enumerate(datasets_list):
            dataset_params = best_params.get(dataset, {})
            p_val = dataset_params.get('p', 0) * 100 if pd.notna(dataset_params.get('p')) else 0
            n_val = int(dataset_params.get('n', 0)) if pd.notna(dataset_params.get('n')) else 0
            pp_val = dataset_params.get('pp', 0) * 100 if pd.notna(dataset_params.get('pp')) else 0
            params_str = f" \\hfill \\footnotesize (Sel: {p_val:.0f}\\%, Augs: {n_val}, Pts: {pp_val:.0f}\\%)"

            dataset_display = dataset.replace('_', ' ').title()
            num_cols = 7
            cell_content = f"\\textit{{\\textbf{{{dataset_display} Dataset}}}}{params_str}"
            f.write(f"        \\multicolumn{{{num_cols}}}{{l}}{{{cell_content}}} \\\\\n")

            dataset_data = pivot_table[pivot_table['dataset'] == dataset]
            baseline_scores = dataset_data[dataset_data['strategy'] == 'baseline'].set_index('model')['score']

            for model in models_list:
                model_data = dataset_data[dataset_data['model'] == model]
                if model_data.empty: continue

                model_display = model_abbreviations.get(model, model)
                row = f"        {model_display:<15}"
                baseline_score_for_model = baseline_scores.get(model, 0)

                for strategy in strategies_list:
                    strategy_data = model_data[model_data['strategy'] == strategy]
                    if not strategy_data.empty:
                        score = strategy_data['score'].iloc[0]
                        # --- THE EXACT FIX: Manually create the string with the % sign ---
                        percentage_str = f"{score * 100:.2f}\\%"

                        if strategy != 'baseline' and score > baseline_score_for_model:
                            row += f" & \\textbf{{{percentage_str}}}"
                        else:
                            row += f" & {percentage_str}"
                    else:
                        row += " & --"
                row += " \\\\\n"
                f.write(row)

            if i < len(datasets_list) - 1:
                f.write("        \\midrule\n")

        f.write("        \\bottomrule\n")
        f.write("    \\end{tabular}\n")
        f.write("    }\n")  # Close the resizebox
        f.write("\\end{table}\n")


def create_final_summary_visualizations(results_df: pd.DataFrame, output_dir: str, dataset_name: str):
    """
    Creates and saves 4 summary visualizations for a specific dataset.
    """
    logger.info(f"Generating visualizations for '{dataset_name}' in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    palette = 'viridis'
    if results_df['strategy'].nunique() > 10:
        palette = sns.color_palette("husl", results_df['strategy'].nunique())

    df = results_df.copy()
    df['strategy_display'] = df['strategy'].apply(_clean_strategy_name)

    # Calculate average scores
    avg_results = df.groupby(['model', 'strategy_display'])['score'].mean().reset_index()

    # --- PLOT 1: Overall Model Performance Bar Plot ---
    plt.figure(figsize=(12, 8))
    sns.barplot(x='model', y='score', hue='strategy_display', data=avg_results, palette=palette)
    plt.title(f'Model Performance for {dataset_name.title()} (Optimal Parameters)', fontsize=16)
    plt.ylabel('Mean Weighted F1-Score', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.ylim(0, max(1.0, avg_results['score'].max() * 1.1))
    plt.legend(title='Strategy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_final_model_performance.png'), dpi=300)
    plt.close()

    # --- PLOT 2: Improvement Percentage ---
    if 'Baseline' in avg_results['strategy_display'].unique():
        pivot = avg_results.pivot_table(index='model', columns='strategy_display', values='score')

        improvement_df = pd.DataFrame()
        aug_strategies = [s for s in pivot.columns if s != 'Baseline']

        for strategy in aug_strategies:
            pivot[f'{strategy}'] = (
                    (pivot[strategy] - pivot['Baseline']) / pivot['Baseline'] * 100
            ).fillna(0)

        # Melt for plotting
        improvement_cols = aug_strategies
        if improvement_cols:
            improvement = pivot[improvement_cols].reset_index()
            improvement_df = improvement.melt(id_vars='model', var_name='Strategy', value_name='Improvement (%)')

            plt.figure(figsize=(14, 8))
            ax = sns.barplot(x='model', y='Improvement (%)', hue='Strategy', data=improvement_df, palette=palette)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f%%', fontsize=9, label_type='edge')

            plt.title(f'F1-Score Improvement vs Baseline for {dataset_name.title()}', fontsize=16, pad=20)
            plt.axhline(y=0, color='red', linestyle='--')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{dataset_name}_final_improvement_comparison.png'), dpi=300)
            plt.close()

    # --- PLOT 3: Heatmap ---
    heatmap_data = avg_results.groupby(['model', 'strategy_display'])['score'].mean().unstack()
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlGnBu', linewidths=.5)
    plt.title(f'Mean F1-Score Heatmap for {dataset_name.title()}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_final_strategy_heatmap.png'), dpi=300)
    plt.close()

    # --- PLOT 4: Ranking ---
    ranking = avg_results.groupby('strategy_display')['score'].mean().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(x='strategy_display', y='score', data=ranking, palette=palette)
    plt.title(f'Overall Strategy Ranking for {dataset_name.title()}', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_final_strategy_ranking.png'), dpi=300)
    plt.close()


def create_final_summary_reports(results_dir: str, datasets_to_process: Optional[List[str]] = None):
    """
    Main entry point for generating reports.
    1. Loads all results.
    2. Identifies optimal parameters per dataset.
    3. Filters data to those parameters.
    4. Generates LaTeX table and Visualizations.
    """
    logger.info("Starting report generation...")

    # 1. Load Data
    master_df = _load_and_parse_results(results_dir, datasets_to_process)
    if master_df.empty:
        logger.error("No data available to report.")
        return

    # 2. Find Best Parameters per Dataset
    aug_only_df = master_df[master_df['strategy'] != 'baseline'].copy()
    best_params_per_dataset = {}

    if not aug_only_df.empty:
        # Average across models and seeds to find the best generic augmentation setting
        param_performance = aug_only_df.groupby(
            ['dataset', 'proportion', 'n_aug_trajs', 'points_proportion']
        )['score'].mean().reset_index()

        best_indices = param_performance.groupby('dataset')['score'].idxmax()
        best_params_df = param_performance.loc[best_indices]

        for _, row in best_params_df.iterrows():
            best_params_per_dataset[row['dataset']] = {
                'p': row['proportion'],
                'n': row['n_aug_trajs'],
                'pp': row['points_proportion']
            }
            logger.info(
                f"Optimal params for {row['dataset']}: "
                f"Select={row['proportion']:.1f}, N={int(row['n_aug_trajs'])}, Pts={row['points_proportion']:.1f}"
            )

    # 3. Filter Data for Final Report
    dfs_to_keep = []

    # Always keep baseline
    dfs_to_keep.append(master_df[master_df['strategy'] == 'baseline'])

    # Keep specific optimal augmented rows
    for dataset, params in best_params_per_dataset.items():
        subset = master_df[
            (master_df['dataset'] == dataset) &
            (master_df['proportion'] == params['p']) &
            (master_df['n_aug_trajs'] == params['n']) &
            (master_df['points_proportion'] == params['pp'])
            ]
        dfs_to_keep.append(subset)

    final_df = pd.concat(dfs_to_keep, ignore_index=True)
    # If only two seeds > remove dfs_to_keep_.append(subset) and apply:
    # final_df = master_df

    # 4. Generate Outputs
    report_dir = os.path.join(results_dir, 'analysis', 'reports')
    img_base_dir = os.path.join(results_dir, 'analysis', 'images')

    os.makedirs(report_dir, exist_ok=True)

    # LaTeX Table
    tex_path = os.path.join(report_dir, 'thesis_final_results_table.tex')
    _generate_latex_table(final_df, best_params_per_dataset, tex_path)
    logger.info(f"LaTeX table saved to {tex_path}")

    # Visualizations
    for dataset in final_df['dataset'].unique():
        dataset_df = final_df[final_df['dataset'] == dataset]
        img_dir = os.path.join(img_base_dir, dataset) # Subfolder per dataset
        create_final_summary_visualizations(dataset_df, img_dir, dataset)

    logger.info("Report generation complete.")