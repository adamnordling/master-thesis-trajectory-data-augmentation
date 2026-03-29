# Trajectory Data Augmentation

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Thesis_Complete-orange)

A modular, configuration-driven Python system for generating, evaluating, and comparing augmented trajectory datasets to improve classification models. This project implements multiple selection strategies (Random, Outlierness, Diversity, Representativeness, Uncertainty) and optimizes them using **Optuna**.

---

## 📂 Project Structure

This project follows a strict Source Layout (`src/`) for modularity and scalability.

```text
trajectory-augmentation/
├── config/                 # YAML Configuration files (The Control Center)
│   ├── base.yaml           # Global paths and seed settings
│   ├── models.yaml         # Hyperparameter grids for ML models
│   └── strategies.yaml     # Parameters for selection algorithms
├── data/                   # Data Storage (Git Ignored)
│   ├── raw/                # Immutable original CSVs
│   ├── augmented/          # Generated Feather files per seed
│   └── output/             # Results, Logs, and Visualizations
├── src/                    # Source Code
│   ├── core/               # Pure logic (Geometry, Features, Outlier Math)
│   ├── evaluation/         # Model Training & Reporting
│   ├── generators/         # Augmentation Logic
│   ├── pipeline/           # Orchestration & Workers
│   ├── strategies/         # Selection Strategy Implementations
│   └── utils/              # I/O, Logging, Config Parsers
├── tests/                  # Unit Tests
├── pyproject.toml          # Project dependencies and build config
└── main.py                 # CLI Entry Point
```

---

## 🚀 Setup & Installation

### 1. Prerequisite: Get Data & Clone Repository
```bash
git clone git@gitlab.lnu.se:an223ym/master-thesis-trajectory-data-augmentation.git
```

Place your raw trajectory `.csv` files into the `data/raw/` directory.
The system expects columns: `tid` (Trajectory ID), `lat`, `lon`, `time`, `label`.

💾 **Dataset Download**: You can download the exact experimental datasets used in this thesis via [**this Google Drive link**](https://drive.google.com/drive/folders/16TbxggA4w11NA7CjXRrJO0q5MNJVD0ho?usp=sharing).

### 2. Install Package
This project uses `pyproject.toml` for modern dependency management. You can set it up automatically via the included Makefile (Linux/Mac/WSL) or manually (Windows).


**Method A: Automated Setup (Mac / Linux / Windows WSL) - Recommended**
```bash
# Automatically creates a virtual environment and installs all dependencies
make install

# Enter the virtual environment shell
make start
```
**Method B: Manual Setup (Windows Native)**
```bash
# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate

# Upgrade pip and install the project with dev dependencies
python -m pip install --upgrade pip

# Install the project with dev dependencies (for testing, linting, etc.)
pip install -e ".[dev]"
```

## 🛠️ Code Quality & Maintenance
This project enforces strict coding standards using **Ruff** (formatting/linting) and **Mypy** (static type checking). If you are using the Makefile, you can automatically clean and format the codebase:
```bash
make format      # Auto-fixes imports, syntax, and formatting using Ruff
make lint        # Runs strict type checking (Mypy) and linter rules
make clean-python # Removes all __pycache__ and temp build files
```

---

## ⚙️ Configuration

Control the experiment without touching code using the files in `config/`:

*   **`base.yaml`**: Set your random seeds, CPU limits, and file paths.
*   **`strategies.yaml`**: Toggle active strategies and tune algorithm parameters (e.g., K-Means clusters).
*   **`models.yaml`**: Define the Search Space for XGBoost, RandomForest, MLP, etc.

---

## 🖥️ Usage

The system is controlled via `main.py`. It supports two modes: **Automatic Tuning** (Optuna) and **Manual Execution**.

You can run these via standard Python commands, or by using the `Makefile` shortcuts.

### 1. Automatic Hyperparameter Tuning (Recommended)
This runs the full loop: checks cache, optimizes parameters via Optuna, does the full augment and extraction for all datasets in `/data/raw/` and generates final reports.

```bash
make run
# Or manually: python main.py
```

### 2. Manual Pipeline Steps
You can run specific stages of the pipeline individually for debugging or fine-tuning.

**Run a Specific Dataset only:**
```bash
make run-dataset DS=foxes
# Or manually: python main.py --datasets foxes
```

**Generate Reports/Analyze Only::**
```bash
make run-analyze
# Or manually: python main.py --analyze
```

**Run Augmentation with Custom Settings (Manual mode):**
```bash
python main.py --augment --datasets foxes --strategies diversity --prop 0.4 --n-aug 5
```

**Quick Test Mode (Runs only 2 seeds):**
```bash
make test-run
# Or manually: python main.py --test
```

### 3. Cleaning Data

The `Makefile` provides granular commands to clean your workspace without deleting everything:
```bash
make clean-eval      # Deletes model logs/results, but KEEPS the heavy augmented datasets
make clean-aug       # Deletes generated augmented data, keeps original raw data
make clean-dataset DS=foxes  # Wipes all generated data/results for just ONE dataset
make clean-all       # Wipes everything generated by the pipeline
```
---

## 📊 Outputs

All artifacts are saved in `data/output/`:
* **`optimization/history/`**: CSV files containing the F1-scores for all Optuna trials and baseline runs.
* **`model_states/`**: Cached optimal hyperparameters for baseline models.
* **`analysis/images/`**: Auto-generated performance bar charts, heatmaps, and rankings.
* **`analysis/reports/`**: Auto-generated LaTeX tables formatted for academic publication containing paired t-test p-values.
* **`performance/`**: Auto-generated hardware performance audits and step-by-step execution times.
---

## Experimental Datasets & Runtimes

The full experimental pipeline was run on four distinct datasets. This section provides an overview of their characteristics and the approximate computational cost to reproduce the results on the benchmark system.

**Benchmark System Specifications:**
- **Processor:** Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz
- **Physical Cores:** 8
- **Parallel Workers Used:** 6
- **RAM:** 32 GB
- **Storage:** Kingston SFYRS 1000G SSD

| Dataset | Size (Disk) | ⏱️ Total Time | ⚙️ Baseline Tuning | 🔬 Optuna Tuning |
| :--- | :--- | :--- | :--- | :--- |
| **AIS Subset** | ~165 GB | **~24.2 hours** | ~2.6 hours | ~21.5 hours |
| **Starkey** | ~20 GB | **~3.9 hours** | ~36.5 minutes | ~3.3 hours |
| **Foxes** | ~21 GB | **~2.2 hours** | ~15.1 minutes | ~1.9 hours |
| **Car Traffic** | ~2.5 GB | **~1.2 hours** | ~17.6 minutes | ~55.8 minutes |

*All runtimes are approximate and represent the total wall-clock time for a full, end-to-end run (20 seeds, 27 Optuna trials per dataset).*