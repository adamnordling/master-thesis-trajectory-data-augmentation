# Trajectory Data Augmentation

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
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

### 1. Prerequisite: Get Data & Clone
```bash
    # Clone the repository
    git clone git@gitlab.lnu.se:an223ym/master-thesis-trajectory-data-augmentation.git
    cd trajectory-data-augmentation
  ```

Place your raw trajectory `.csv` files into the `data/raw/` directory.
The system expects columns: `tid` (Trajectory ID), `lat`, `lon`, `time`, `label`.

To get the datasets I used, use [**this Google Drive link**](https://drive.google.com/drive/folders/16TbxggA4w11NA7CjXRrJO0q5MNJVD0ho?usp=sharing) to download those.

### 2. Install Package
We use `pyproject.toml` for modern dependency management. Installing in "editable" mode (`-e`) ensures the `src` package is available throughout the project.

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the project and dependencies in editable mode
pip install -e .

# Install the project with dev dependencies (for testing, linting, etc.)
pip install -e ".[dev]"
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

### 1. Automatic Hyperparameter Tuning (Recommended)
This runs the full loop: checks cache, optimizes parameters via Optuna, does the full augment and extraction for all datasets in /data/raw/ and generates final reports.

```bash
python main.py
```

### 2. Manual Pipeline Steps
You can run specific stages of the pipeline individually for debugging or fine-tuning.

**Prepare Data (Split & Clean):**
```bash
python main.py --prepare
```

**Run Augmentation with Custom Settings:**
```bash
python main.py --augment --datasets foxes --strategies diversity --prop 0.4 --n-aug 5
```

**Run a Specific Dataset only:**
```bash
python main.py --datasets foxes
```

**Generate Reports/Analyze Only:**
```bash
python main.py --datasets foxes --report --analyze
```

**Quick Test Mode (Runs only 2 seeds):**
```bash
python main.py --datasets foxes --test
```

---

## 📊 Outputs

All artifacts are saved in `data/output/`:
*   **`augmented/`**: All augmented/merged/extracted .feather files for every seed.
*   **`output/`**: Output of analysis images, LaTex tables and model states.
*   **`logs/`**: Execution logs for debugging.

---

## Experimental Datasets & Runtimes

The full experimental pipeline was run on four distinct datasets. This section provides an overview of their characteristics and the approximate computational cost to reproduce the results on the benchmark system.

**Benchmark System Specifications:**
- **Processor:** Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz (Values inferred from "Family 6 Model 158 Stepping 12")
- **Physical Cores:** 8
- **Parallel Workers Used:** 6
- **RAM:** 32 GB

| Dataset | Size (Disk) | ⏱️ Total Time | ⚙️ Baseline Tuning | 🔬 Optuna Tuning |
| :--- | :--- | :--- | :--- | :--- |
| **AIS Subset** | ~165 GB | **~26.6 hours** | ~2.5 hours | ~24.0 hours |
| **Starkey** | ~20 GB | **~4.4 hours** | ~32.5 minutes | ~3.8 hours |
| **Foxes** | ~21 GB | **~1.9 hours** | ~11.3 minutes | ~1.7 hours |
| **Car Traffic** | ~2.5 GB | **~57 minutes** | ~11.8 minutes | ~44.6 minutes |

*All runtimes are approximate and represent the total wall-clock time for a full, end-to-end run (20 seeds, 27 Optuna trials per dataset).*
