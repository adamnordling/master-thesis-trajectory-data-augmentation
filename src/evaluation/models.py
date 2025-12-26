import ast
import logging
import os
import random
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.preparation import prepare_data

# Initialize logger
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Inconsistent values: penalty=l1")

def set_reproducibility(seed: int) -> None:
    """
    Set seeds for all relevant libraries to ensure full reproducibility.

    Ensures that Python, NumPy, and random state are locked to the specific seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_model_definitions(
    seed: int, model_config: Dict[str, Any], n_jobs: int = 1
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Instantiates model objects and builds hyperparameter grids from the YAML config.

    Args:
        seed: Random seed for initialization.
        model_config: Dictionary loaded from config/models.yaml.
        n_jobs: Number of cores to use *per model*. Usually 1 if using GridSearchCV.

    Returns:
        models: Dictionary of instantiated model objects.
        param_grids: Dictionary of parameter grids formatted for GridSearchCV.
    """
    # 1. Instantiate Models (Architecture)
    # Note: We keep the architecture definitions here (Pipelines, scalers)
    # but the hyperparameters for tuning come from the config.
    models = {
        "RandomForest": RandomForestClassifier(random_state=seed, n_jobs=n_jobs),
        "XGBoost": xgb.XGBClassifier(random_state=seed, n_jobs=n_jobs, eval_metric="mlogloss"),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    max_iter=500,
                    early_stopping=True,
                    n_iter_no_change=15,
                    random_state=seed,
                ),
            ),
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=10000, random_state=seed, n_jobs=n_jobs),
            ),
        ]),
    }

    # 2. Build Grids from Config
    # Map clean YAML keys to specific Scikit-Learn parameter names for Pipelines.
    raw_grids = model_config.get("models", {})
    param_grids: Dict[str, Any] = {}

    # RandomForest (Direct mapping)
    if "RandomForest" in raw_grids:
        param_grids["RandomForest"] = raw_grids["RandomForest"]

    # XGBoost (Direct mapping)
    if "XGBoost" in raw_grids:
        param_grids["XGBoost"] = raw_grids["XGBoost"]

    # MLP (Pipeline: prefix keys with 'classifier__')
    if "MLP" in raw_grids:
        param_grids["MLP"] = {f"classifier__{k}": v for k, v in raw_grids["MLP"].items()}

    # LogisticRegression (Pipeline: prefix keys with 'classifier__')
    if "LogisticRegression" in raw_grids:
        param_grids["LogisticRegression"] = {f"classifier__{k}": v for k, v in raw_grids["LogisticRegression"].items()}

    return models, param_grids


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    feature_type_name: str,
    cached_params: pd.DataFrame,
    model_config: Dict[str, Any],
    n_jobs_gridsearch: int = -1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Trains all defined models on the provided data.

    Handles baseline tuning via GridSearch or loads cached parameters for augmented runs.
    """
    set_reproducibility(seed)

    results: List[Dict[str, Any]] = []
    new_params: List[Dict[str, Any]] = []

    # 1. Prepare Data
    X_train, y_train, _ = prepare_data(train_df)
    X_test, y_test, _ = prepare_data(test_df)

    # Align columns to ensure Train and Test have identical feature sets
    common_cols = sorted(list(set(X_train.columns) & set(X_test.columns)))
    X_train, X_test = X_train[common_cols], X_test[common_cols]

    # 2. Get Models & Grids from Config
    models, param_grids = get_model_definitions(seed, model_config, n_jobs=1)

    # 3. Iterate over algorithms
    for model_name, model_instance in models.items():
        try:
            is_augmented = "merged" in feature_type_name
            param_lookup_key = "trajectory_features"

            # Check cache for baseline parameters
            cache_hit = cached_params[
                (cached_params["feature_type"] == param_lookup_key)
                & (cached_params["seed"] == seed)
                & (cached_params["model"] == model_name)
            ]

            final_model = None

            if not cache_hit.empty:
                # Case A: Cache Hit (Use existing baseline params for this seed)
                best_params_str = cache_hit["best_params"].iloc[0]
                best_params = ast.literal_eval(best_params_str)
                final_model = model_instance.set_params(**best_params)
                final_model.fit(X_train, y_train)
                logger.debug(f"Loaded cached params for {model_name} (Seed {seed})")

            elif not is_augmented:
                # Case B: Baseline Tuning (First time tuning for this seed)
                logger.info(f"Tuning {model_name} for seed {seed}...")

                grid = param_grids.get(model_name, {})
                if not grid:
                    logger.warning(f"No grid defined for {model_name}. Using defaults.")

                grid_search = GridSearchCV(
                    estimator=model_instance,
                    param_grid=grid,
                    scoring="f1_weighted",
                    cv=3,
                    n_jobs=n_jobs_gridsearch,
                    verbose=0,
                )
                grid_search.fit(X_train, y_train)
                final_model = grid_search.best_estimator_

                # Store parameters for future augmented runs
                new_params.append({
                    "feature_type": "trajectory_features",
                    "seed": seed,
                    "model": model_name,
                    "best_params": str(grid_search.best_params_),
                })
            else:
                # Case C: Safety check (Augmented run without prior baseline params)
                logger.error(f"Missing baseline params for {model_name} (Seed {seed}). Skipping.")
                continue

            # 4. Predict and Score
            y_pred = final_model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            results.append({
                "feature_type": feature_type_name,
                "seed": seed,
                "model": model_name,
                "score": f1,
            })

        except Exception as e:
            logger.error(f"Error training {model_name} on seed {seed}: {e}")
            continue

    return results, new_params
