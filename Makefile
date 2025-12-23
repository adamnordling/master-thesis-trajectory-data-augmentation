# --- Variables ---
PYTHON = python
PIP    = pip
# If you don't provide F=..., it cleans the entire 'src' directory by default
F      ?= src

.PHONY: help install setup tidy clean run-full run-smoke run-debug

help:
	@echo "================================================================"
	@echo "      TRAJECTORY AUGMENTATION PIPELINE - DEVELOPER TOOLS"
	@echo "================================================================"
	@echo "ENVIRONMENT"
	@echo "  make start        - Start virtual environment"
	@echo "  make install       - Install project + dev tools (Ruff, Mypy, etc)"
	@echo "  make setup         - Full clean install (removes old env first)"
	@echo ""
	@echo "CODE QUALITY (THE GOLDEN STANDARD)"
	@echo "  make tidy          - The 'Super-Clean': Fixes types, linting, & format"
	@echo "                       Usage: make tidy F=src/core/features.py"
	@echo "  make clean         - Remove temporary caches and build artifacts"
	@echo ""
	@echo "PIPELINE EXECUTION"
	@echo "  make run-full      - Run the complete Optuna optimization"
	@echo "  make run-smoke     - Quick parallel test to verify infrastructure"
	@echo "  make run-debug     - Sequential test with verbose errors"
	@echo "================================================================"

# --- 1. Environment Management ---

start:
	powershell -NoExit -Command ". .venv\Scripts\Activate.ps1"

install:
	$(PIP) install -e ".[dev]"

setup: clean
	$(PYTHON) -m venv .venv
	@echo "Virtual environment created. Run 'source .venv/bin/activate' or similar."

# --- 2. The "Golden Standard" Clean (Min-Maxed) ---

tidy:
	@echo "--- Step 1: Injecting Type Hints (Autotyping) ---"
	-python -m autotyping $(F) --none-return --scalar-return --int-param --float-param --str-param

	@echo "--- Step 2: Fixing Linting & Docstrings (Ruff) ---"
	ruff check --fix --unsafe-fixes $(F)

	@echo "--- Step 3: Global Formatting (Ruff) ---"
	ruff format $(F)

	@echo "--- Step 4: Static Analysis Check (Mypy) ---"
	-mypy $(F)
	@echo "✨ File(s) are now at the Gold Standard."

# --- 3. Cleaning Artifacts ---

clean:
	@echo "Cleaning caches and data outputs..."
	rm -rf `find . -type d -name "__pycache__"`
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.log" -delete
	rm -rf data/augmented/* data/output/*

# --- 4. Pipeline Execution Scenarios ---

run-full:
	$(PYTHON) main.py

run-smoke:
	$(PYTHON) main.py --test

run-debug:
	$(PYTHON) main.py --test --no-parallel