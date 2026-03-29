import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Internal imports from our new core structure
from src.core.outliers import detect_outliers_dbos, find_average_distance
from src.strategies.base import TrajectorySelectionMethod

logger = logging.getLogger(__name__)


class RandomTrajectorySelection(TrajectorySelectionMethod):
    """Selects trajectories purely at random, respecting class balance.
    Parameters: None specific.
    """

    def __init__(self, proportion: float, seed: int = 1415, params: dict[str, Any] | None = None) -> None:
        """Initialize RandomTrajectorySelection."""
        super().__init__(proportion, seed)
        self.params = params or {}

    def select(self, trajectory_feats_df: pd.DataFrame) -> list[str]:
        """Selects trajectories at random while maintaining class balance."""
        unique_tids = trajectory_feats_df["tid"].unique()
        unique_labels = trajectory_feats_df["label"].unique()
        selected_tids: list[str] = []

        rng = np.random.default_rng(self.seed)

        total_to_select = int(len(unique_tids) * self.proportion)
        total_trajectories = len(unique_tids)

        for label in unique_labels:
            label_tids = trajectory_feats_df[trajectory_feats_df["label"] == label]["tid"].unique()

            if len(label_tids) == 0:
                continue

            # Calculate proportional allocation
            label_proportion = len(label_tids) / total_trajectories
            num_to_select = int(total_to_select * label_proportion)
            num_to_select = min(num_to_select, len(label_tids))

            if num_to_select > 0:
                # Use to_numpy() to ensure standard array support for rng.choice
                selected = rng.choice(np.asarray(label_tids), num_to_select, replace=False)
                selected_tids.extend(selected.astype(str))

        logger.info(f"Random Strategy selected {len(selected_tids)} trajectories.")
        return selected_tids


class SelectTrajectoryBasedOnOutlierness(TrajectorySelectionMethod):
    """Selects the most 'outlier-ish' trajectories using DBOS.

    YAML Configuration Usage:
        outlierness:
            default_radius: 1.0  # Used if average distance calc fails
    """

    def __init__(self, proportion: float, seed: int = 1415, params: dict[str, Any] | None = None) -> None:
        """Initializes the outlierness selector with optional parameters."""
        super().__init__(proportion, seed)
        self.params = params or {}
        self.default_radius = self.params.get("default_radius", 1.0)

    def select(self, trajectory_feats_df: pd.DataFrame) -> list[str]:
        """Selects trajectories with the highest outlier scores based on DBOS, ensuring class balance."""
        df = (
            trajectory_feats_df.reset_index()
            if "tid" not in trajectory_feats_df.columns
            else trajectory_feats_df.copy()
        )

        unique_tids = df["tid"].unique()
        unique_labels = [label for label in df["label"].unique() if pd.notna(label)]
        selected_tids: list[str] = []

        total_to_select = int(len(unique_tids) * self.proportion)
        num_per_label = max(1, total_to_select // len(unique_labels)) if unique_labels else 0

        for label in unique_labels:
            label_df = df[df["label"] == label].copy()
            if label_df.empty:
                continue

            tids = label_df["tid"].to_numpy(dtype=str)
            feature_cols = [c for c in label_df.columns if c not in ["tid", "label", "time"]]
            features = label_df[feature_cols].to_numpy(dtype=float)

            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            if features.std() < 1e-9:
                logger.warning(f"Label {label} has zero variance. Falling back to random selection.")
                rng = np.random.default_rng(self.seed)
                cnt = min(num_per_label, len(tids))
                if cnt > 0:
                    selected_tids.extend(rng.choice(tids, cnt, replace=False))
                continue

            avg_dist = find_average_distance(features, seed=self.seed)

            if avg_dist <= 0:
                avg_dist = self.default_radius

            try:
                results = detect_outliers_dbos(features, d=avg_dist)
                scores = results["scores"].flatten()
                sorted_indices = np.argsort(scores)[::-1]

                cnt = min(num_per_label, len(tids))
                top_indices = sorted_indices[:cnt]
                selected_tids.extend(tids[top_indices].tolist())

            except Exception as e:
                logger.error(f"DBOS failed for label {label}: {e}. Skipping.")
                continue

        logger.info(f"Outlierness Strategy selected {len(selected_tids)} trajectories.")
        return selected_tids


class SelectTrajectoryBasedOnDiversityMaximization(TrajectorySelectionMethod):
    """Selects diverse trajectories using K-Means clustering."""

    def __init__(self, proportion: float, seed: int = 1415, params: dict[str, Any] | None = None) -> None:
        """Initializes the diversity maximization selector with optional parameters for clustering."""
        super().__init__(proportion, seed)
        self.params = params or {}
        self.k_clusters = self.params.get("k_clusters", 5)
        self.batch_size = self.params.get("batch_size", 256)

    def select(self, trajectory_feats_df: pd.DataFrame) -> list[str]:
        """Selects trajectories with the highest diversity by clustering and sampling from clusters, ensuring class balance."""
        df = (
            trajectory_feats_df.reset_index()
            if "tid" not in trajectory_feats_df.columns
            else trajectory_feats_df.copy()
        )

        unique_tids = df["tid"].unique()
        unique_labels = [label for label in df["label"].unique() if pd.notna(label)]
        selected_tids: list[str] = []

        total_to_select = max(1, int(len(unique_tids) * self.proportion))
        num_per_label = max(1, total_to_select // len(unique_labels)) if unique_labels else 0

        for label in unique_labels:
            label_df = df[df["label"] == label]
            if len(label_df) < 2:
                continue

            tids = label_df["tid"].to_numpy(dtype=str)
            feature_cols = [c for c in label_df.columns if c not in ["tid", "label", "time"]]
            features = label_df[feature_cols].fillna(0).to_numpy(dtype=float)

            effective_k = min(self.k_clusters, len(features))

            if effective_k < 2:
                selected_tids.extend(tids[: min(num_per_label, len(tids))].tolist())
                continue

            if len(features) > 1000:
                kmeans = MiniBatchKMeans(
                    n_clusters=effective_k,
                    random_state=self.seed,
                    batch_size=self.batch_size,
                )
            else:
                kmeans = KMeans(n_clusters=effective_k, random_state=self.seed, n_init=10)

            try:
                clusters = kmeans.fit_predict(features)
                cluster_df = pd.DataFrame({"tid": tids, "cluster": clusters})

                for c_id in range(effective_k):
                    c_samples = cluster_df[cluster_df["cluster"] == c_id]
                    if c_samples.empty:
                        continue

                    c_prop = len(c_samples) / len(cluster_df)
                    c_select = max(1, int(num_per_label * c_prop))

                    rng = np.random.default_rng(self.seed + c_id)
                    chosen = rng.choice(
                        c_samples["tid"].to_numpy(dtype=str),
                        min(c_select, len(c_samples)),
                        replace=False,
                    )
                    selected_tids.extend(chosen.tolist())

            except Exception as e:
                logger.error(f"Clustering failed for label {label}: {e}")
                continue

        logger.info(f"Diversity Strategy selected {len(selected_tids)} trajectories.")
        return selected_tids


class SelectTrajectoryBasedOnRepresentativeness(TrajectorySelectionMethod):
    """Selects most representative (central) trajectories using Z-scores."""

    def __init__(self, proportion: float, seed: int = 1415, params: dict[str, Any] | None = None) -> None:
        """Initializes the representativeness selector with optional parameters (currently unused but can be extended)."""
        """Initializes the representativeness selector."""
        super().__init__(proportion, seed)
        self.params = params or {}

    def select(self, trajectory_feats_df: pd.DataFrame) -> list[str]:
        """Selects trajectories that are most representative (closest to mean) based on Z-scores, ensuring class balance."""
        df = (
            trajectory_feats_df.reset_index()
            if "tid" not in trajectory_feats_df.columns
            else trajectory_feats_df.copy()
        )

        unique_tids = df["tid"].unique()
        unique_labels = [label for label in df["label"].unique() if pd.notna(label)]
        selected_tids: list[str] = []

        total_to_select = int(len(unique_tids) * self.proportion)
        total_rows = len(df[df["label"].isin(unique_labels)])

        for label in unique_labels:
            label_df = df[df["label"] == label]
            if label_df.empty:
                continue

            num_to_select = int(total_to_select * (len(label_df) / total_rows))
            num_to_select = min(num_to_select, len(label_df))
            if num_to_select == 0:
                continue

            tids = label_df["tid"].to_numpy(dtype=str)
            feature_cols = [c for c in label_df.columns if c not in ["tid", "label", "time"]]
            features = label_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

            mean_vals = features.mean(axis=0)
            std_vals = features.std(axis=0)
            std_vals[std_vals < 1e-9] = 1.0

            z_scores = (features - mean_vals) / std_vals
            rep_scores = z_scores.abs().sum(axis=1).to_numpy()

            sorted_indices = np.argsort(rep_scores)
            top_indices = sorted_indices[:num_to_select]

            selected_tids.extend(tids[top_indices].tolist())

        logger.info(f"Representativeness Strategy selected {len(selected_tids)} trajectories.")
        return selected_tids


class SelectTrajectoryBasedOnUncertainty(TrajectorySelectionMethod):
    """Selects trajectories where a model is most uncertain (Active Learning)."""

    def __init__(self, proportion: float, seed: int = 1415, params: dict[str, Any] | None = None) -> None:
        """Initializes the uncertainty selector with a simple logistic regression model and optional parameters for model configuration."""
        super().__init__(proportion, seed)
        self.params = params or {}
        self.scout_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=self.seed, solver="lbfgs", max_iter=1000),
        )

    def select(self, trajectory_feats_df: pd.DataFrame) -> list[str]:
        """Selects trajectories where the logistic regression model is most uncertain (closest to 0.5 probability), ensuring class balance."""
        df = (
            trajectory_feats_df.reset_index()
            if "tid" not in trajectory_feats_df.columns
            else trajectory_feats_df.copy()
        )

        feature_cols = [c for c in df.columns if c not in ["tid", "label", "time"]]
        features = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).to_numpy(dtype=float)
        labels = df["label"].to_numpy()
        tids = df["tid"].to_numpy(dtype=str)

        if len(np.unique(labels)) < 2:
            logger.warning("Uncertainty selection requires at least 2 classes. Skipping.")
            return []

        logger.info("Uncertainty Strategy: Training scout model...")
        self.scout_model.fit(features, labels)

        predicted_probabilities = self.scout_model.predict_proba(features)
        uncertainty_scores = entropy(predicted_probabilities.T)

        scores_df = pd.DataFrame({"tid": tids, "label": labels, "uncertainty": uncertainty_scores})

        selected_tids: list[str] = []
        for _, group in scores_df.groupby("label"):
            group = group.sort_values(by="uncertainty", ascending=False)
            num_to_select = max(1, int(len(group) * self.proportion))
            selected_tids.extend(group.head(num_to_select)["tid"].tolist())

        logger.info(f"Uncertainty Strategy selected {len(selected_tids)} trajectories.")
        return selected_tids
