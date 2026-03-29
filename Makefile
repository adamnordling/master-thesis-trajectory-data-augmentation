.PHONY: help install start format lint test-run run run-dataset run-analyze clean-python clean-all clean-aug clean-eval clean-dataset check-ds

VENV_DIR = venv

# === OS-SPECIFIC SETUP ===
# Automatically detects if you are on Windows or Linux/Mac and assigns the correct commands.
ifeq ($(OS),Windows_NT)
	# The -u flag forces Unbuffered output so TQDM progress bars render smoothly!
	PYTHON = $(VENV_DIR)/Scripts/python.exe -u
	PIP = $(VENV_DIR)/Scripts/pip.exe
	PYTHON_CMD = python
	START_CMD = cmd /k "$(VENV_DIR)\Scripts\activate.bat"

	# PowerShell is used on Windows to safely handle forward slashes (/) and wildcards (*)
	RM_DIR = powershell -Command Remove-Item -Recurse -Force -ErrorAction Ignore
	RM_FILE = powershell -Command Remove-Item -Force -ErrorAction Ignore
	CLEAN_PYCACHE = powershell -Command "Get-ChildItem -Path . -Include __pycache__,.ruff_cache,.mypy_cache,.pytest_cache -Recurse -Force -Directory -ErrorAction Ignore | Remove-Item -Recurse -Force"
else
	# The -u flag forces Unbuffered output so TQDM progress bars render smoothly!
	PYTHON = $(VENV_DIR)/bin/python -u
	PIP = $(VENV_DIR)/bin/pip
	PYTHON_CMD = python3
	START_CMD = bash -c "source $(VENV_DIR)/bin/activate && exec bash"

	RM_DIR = rm -rf
	RM_FILE = rm -f
	CLEAN_PYCACHE = find . -type d -name "__pycache__" -exec rm -rf {} + && rm -rf .ruff_cache .mypy_cache .pytest_cache
endif

help: ## Show this help message
	@echo "================================================================="
	@echo "        Trajectory Data Augmentation Pipeline - Commands         "
	@echo "================================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Environment & Setup ---

install: ## Create virtual environment and install all dependencies (including dev)
	$(PYTHON_CMD) -m venv $(VENV_DIR)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .[dev]
	@echo "Installation complete! Run 'make start' to enter the environment."

start: ## Spawns a new terminal shell with the virtual environment activated
	@echo "Entering virtual environment... (Type 'exit' to leave)"
	@$(START_CMD)

# --- Code Quality (Linting & Formatting) ---

format: ## Auto-format code using Ruff (fixes imports, spacing, and modernizes syntax)
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

lint: ## Run strict type checking (Mypy) and linter warnings (Ruff)
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy src/

clean-python: ## Remove all Python cache files (__pycache__, .ruff_cache, .mypy_cache)
	@$(CLEAN_PYCACHE)
	@echo "Cleaned all Python cache directories."

# --- Execution ---

# Cross-platform way to ensure the user provided the "DS=" argument
check-ds:
ifndef DS
	$(error Error: Please specify a dataset. Example: make run-dataset DS=car_traffic)
endif

test-run: ## Run the pipeline in test mode (only uses 2 seeds for quick debugging)
	$(PYTHON) main.py --test

run: ## Run the full automatic pipeline on all datasets
	$(PYTHON) main.py

run-dataset: check-ds ## Run the pipeline for a specific dataset (Usage: make run-dataset DS=car_traffic)
	$(PYTHON) main.py --datasets $(DS)

run-analyze: ## Run ONLY the final statistical analysis and LaTeX reporting
	$(PYTHON) main.py --analyze

# --- Granular Cleaning (Data & Outputs) ---
# Note: The '-' before $(RM_DIR) tells Make to ignore the error if the folder is already deleted.

clean-all: clean-python ## CAUTION: Delete ALL augmented data and ALL evaluation outputs
	-$(RM_DIR) data/augmented/*
	-$(RM_DIR) data/output/*
	@echo "Cleaned all augmented data and outputs."

clean-aug: ## Delete ONLY the augmented data (.feather files), keep evaluation results
	-$(RM_DIR) data/augmented/*
	@echo "Cleaned augmented data. Raw data is untouched."

clean-eval: ## Delete ONLY the model evaluations, optimization history, and reports. Keeps augmented data!
	-$(RM_DIR) data/output/*
	@echo "Cleaned evaluation outputs. You can now re-evaluate without re-augmenting."

clean-dataset: check-ds ## Delete ALL generated data/results for a specific dataset (Usage: make clean-dataset DS=car_traffic)
	# 1. Remove the heavy augmented data
	-$(RM_DIR) data/augmented/$(DS)
	# 2. Remove the analysis artifacts (Images and LaTeX)
	-$(RM_DIR) data/output/analysis/images/$(DS)
	-$(RM_DIR) data/output/analysis/reports/$(DS)
	# 3. Remove the Optimization detailed results
	-$(RM_DIR) data/output/optimization/details/$(DS)
	# 4. Remove the Optuna Database (This is what tells Optuna to start over)
	-$(RM_FILE) data/output/optuna_$(DS).db
	# 5. Remove the Model State caches (Baseline hyperparameters)
	-$(RM_FILE) data/output/model_states/$(DS)_best_params.csv
	# 6. Remove all History files (Baseline and ALL Optuna trials)
	-$(RM_FILE) data/output/optimization/history/$(DS)_baseline_tuning.csv
	-$(RM_FILE) data/output/optimization/history/$(DS)_p*.csv
	# 7. Remove any performance audits related to this dataset
	-$(RM_FILE) data/output/performance/*$(DS)*_performance_audit.txt
	@echo "Surgically cleaned all generated data, databases, and history for $(DS)."