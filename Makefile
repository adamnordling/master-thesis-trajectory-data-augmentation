.PHONY: help install start format lint test-run run run-dataset run-analyze clean-python clean-all clean-aug clean-eval clean-dataset

VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

help: ## Show this help message
	@echo "================================================================="
	@echo "        Trajectory Data Augmentation Pipeline - Commands         "
	@echo "================================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Environment & Setup ---

install: ## Create virtual environment and install all dependencies (including dev)
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]
	@echo "Installation complete! Run 'make start' to enter the environment."

start: ## Spawns a new terminal shell with the virtual environment activated
	@echo "Entering virtual environment... (Type 'exit' to leave)"
	@bash -c "source $(VENV_DIR)/bin/activate && exec bash"

# --- Code Quality (Linting & Formatting) ---

format: ## Auto-format code using Ruff (fixes imports, spacing, and modernizes syntax)
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

lint: ## Run strict type checking (Mypy) and linter warnings (Ruff)
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy src/

clean-python: ## Remove all Python cache files (__pycache__, .ruff_cache, .mypy_cache)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .ruff_cache .mypy_cache .pytest_cache
	@echo "Cleaned all Python cache directories."

# --- Execution ---

test-run: ## Run the pipeline in test mode (only uses 2 seeds for quick debugging)
	$(PYTHON) main.py --test

run: ## Run the full automatic pipeline on all datasets
	$(PYTHON) main.py

run-dataset: ## Run the pipeline for a specific dataset (Usage: make run-dataset DS=car_traffic)
	@if [ -z "$(DS)" ]; then echo "Error: Please specify a dataset. Example: make run-dataset DS=car_traffic"; exit 1; fi
	$(PYTHON) main.py --datasets $(DS)

run-analyze: ## Run ONLY the final statistical analysis and LaTeX reporting
	$(PYTHON) main.py --analyze

# --- Granular Cleaning (Data & Outputs) ---

clean-all: clean-python ## CAUTION: Delete ALL augmented data and ALL evaluation outputs
	rm -rf data/augmented/*
	rm -rf data/output/*
	@echo "Cleaned all augmented data and outputs."

clean-aug: ## Delete ONLY the augmented data (.feather files), keep evaluation results
	rm -rf data/augmented/*
	@echo "Cleaned augmented data. Raw data is untouched."

clean-eval: ## Delete ONLY the model evaluations, optimization history, and reports. Keeps augmented data!
	rm -rf data/output/*
	@echo "Cleaned evaluation outputs. You can now re-evaluate without re-augmenting."

clean-dataset: ## Delete ALL generated data/results for a specific dataset (Usage: make clean-dataset DS=car_traffic)
	@if [ -z "$(DS)" ]; then echo "Error: Please specify a dataset. Example: make clean-dataset DS=car_traffic"; exit 1; fi
	rm -rf data/augmented/$(DS)
	rm -rf data/output/optimization/details/$(DS)
	rm -rf data/output/analysis/images/$(DS)
	rm -rf data/output/analysis/reports/$(DS)
	rm -f data/output/model_states/$(DS)_best_params.csv
	rm -f data/output/optimization/history/$(DS)_baseline_tuning.csv
	@echo "Cleaned all generated data and results for $(DS)."