import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from typing import List, Dict, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import entropy

# Internal imports from our new core structure
from src.core.outliers import detect_outliers_dbos, find_average_distance
from src.strategies.base import TrajectorySelectionMethod

logger = logging.getLogger(__name__)


class RandomTrajectorySelection(TrajectorySelectionMethod):
    """
    Selects trajectories purely at random, respecting class balance.
    Parameters: None specific.
    """

    def __init__(self, proportion: float, seed: int = 1415, params: Optional[Dict[str, Any]] = None):
        super().__init__(proportion, seed)
        # Random selection doesn't use extra params, but we accept the arg for consistency
        self.params = params or {}

    def select(self, trajectory_feats_df: pd.DataFrame) -> List[str]:
        unique_tids = trajectory_feats_df['tid'].unique()
        unique_labels = trajectory_feats_df['label'].unique()
        selected_tids = []

        if self.seed is not None:
            np.random.seed(self.seed)

        total_to_select = int(len(unique_tids) * self.proportion)
        total_trajectories = len(unique_tids)

        for label in unique_labels:
            label_tids = trajectory_feats_df[trajectory_feats_df['label'] == label]['tid'].unique()

            if len(label_tids) == 0:
                continue

            # Calculate proportional allocation
            label_proportion = len(label_tids) / total_trajectories
            num_to_select = int(total_to_select * label_proportion)
            num_to_select = min(num_to_select, len(label_tids))

            if num_to_select > 0:
                selected = np.random.choice(label_tids, num_to_select, replace=False)
                selected_tids.extend(selected)

        logger.info(f"Random Strategy selected {len(selected_tids)} trajectories.")
        return selected_tids


class SelectTrajectoryBasedOnOutlierness(TrajectorySelectionMethod):
    """
    Selects the most 'outlier-ish' trajectories using DBOS.

    YAML Configuration Usage:
        outlierness:
            default_radius: 1.0  # Used if average distance calc fails
    """

    def __init__(self, proportion: float, seed: int = 1415, params: Optional[Dict[str, Any]] = None):
        super().__init__(proportion, seed)
        self.params = params or {}
        # Load defaults from config or fallback to hardcoded
        self.default_radius = self.params.get('default_radius', 1.0)

    def select(self, trajectory_feats_df: pd.DataFrame) -> List[str]:
        # Handle index vs column for 'tid'
        df = trajectory_feats_df.reset_index() if 'tid' not in trajectory_feats_df.columns else trajectory_feats_df.copy()

        unique_tids = df['tid'].unique()
        unique_labels = [l for l in df['label'].unique() if pd.notna(l)]
        selected_tids = []

        total_to_select = int(len(unique_tids) * self.proportion)
        num_per_label = max(1, total_to_select // len(unique_labels)) if unique_labels else 0

        for label in unique_labels:
            label_df = df[df['label'] == label].copy()
            if label_df.empty: continue

            tids = label_df['tid'].values
            # Drop non-feature columns
            feature_cols = [c for c in label_df.columns if c not in ['tid', 'label', 'time']]
            features = label_df[feature_cols].values

            # Handle NaNs/Infs
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Check for variance
            if features.std() < 1e-9:
                logger.warning(f"Label {label} has zero variance. Falling back to random selection.")
                np.random.seed(self.seed)
                cnt = min(num_per_label, len(tids))
                if cnt > 0:
                    selected_tids.extend(np.random.choice(tids, cnt, replace=False))
                continue

            # 1. Calculate Average Distance
            avg_dist = find_average_distance(features)

            # Use configured default if calculation fails
            if avg_dist <= 0:
                logger.debug(f"Average distance calculation failed/zero. Using default_radius={self.default_radius}")
                avg_dist = self.default_radius

            # 2. Run DBOS (Outlier Detection)
            try:
                results = detect_outliers_dbos(features, d=avg_dist)
                scores = results['scores'].flatten()

                # Sort by score descending (Most outlier-ish first)
                sorted_indices = np.argsort(scores)[::-1]

                # Select top N
                cnt = min(num_per_label, len(tids))
                top_indices = sorted_indices[:cnt]
                selected_tids.extend(tids[top_indices])

            except Exception as e:
                logger.error(f"DBOS failed for label {label}: {e}. Skipping.")
                continue

        logger.info(f"Outlierness Strategy selected {len(selected_tids)} trajectories.")
        return selected_tids


class SelectTrajectoryBasedOnDiversityMaximization(TrajectorySelectionMethod):
    """
    Selects diverse trajectories using K-Means clustering.

    YAML Configuration Usage:
        diversity:
            k_clusters: 5
            batch_size: 256
    """

    def __init__(self, proportion: float, seed: int = 1415, params: Optional[Dict[str, Any]] = None):
        super().__init__(proportion, seed)
        self.params = params or {}
        # Load settings from YAML config (or defaults)
        self.k_clusters = self.params.get('k_clusters', 5)
        self.batch_size = self.params.get('batch_size', 256)

    def select(self, trajectory_feats_df: pd.DataFrame) -> List[str]:
        df = trajectory_feats_df.reset_index() if 'tid' not in trajectory_feats_df.columns else trajectory_feats_df.copy()

        unique_tids = df['tid'].unique()
        unique_labels = [l for l in df['label'].unique() if pd.notna(l)]
        selected_tids = []

        total_to_select = max(1, int(len(unique_tids) * self.proportion))
        num_per_label = max(1, total_to_select // len(unique_labels)) if unique_labels else 0

        for label in unique_labels:
            label_df = df[df['label'] == label]
            if len(label_df) < 2: continue

            tids = label_df['tid'].values
            feature_cols = [c for c in label_df.columns if c not in ['tid', 'label', 'time']]
            features = label_df[feature_cols].fillna(0).values

            effective_k = min(self.k_clusters, len(features))

            if effective_k < 2:
                # Not enough data for clustering
                selected_tids.extend(tids[:min(num_per_label, len(tids))])
                continue

            # Use MiniBatchKMeans for large datasets
            if len(features) > 1000:
                kmeans = MiniBatchKMeans(
                    n_clusters=effective_k,
                    random_state=self.seed,
                    batch_size=self.batch_size
                )
            else:
                kmeans = KMeans(
                    n_clusters=effective_k,
                    random_state=self.seed,
                    n_init=10
                )

            try:
                clusters = kmeans.fit_predict(features)
                cluster_df = pd.DataFrame({'tid': tids, 'cluster': clusters})

                # Stratified selection from clusters
                for c_id in range(effective_k):
                    c_samples = cluster_df[cluster_df['cluster'] == c_id]
                    if c_samples.empty: continue

                    # Allocate based on cluster size
                    c_prop = len(c_samples) / len(cluster_df)
                    c_select = max(1, int(num_per_label * c_prop))

                    # Random select within cluster
                    np.random.seed(self.seed + c_id)
                    chosen = np.random.choice(
                        c_samples['tid'].values,
                        min(c_select, len(c_samples)),
                        replace=False
                    )
                    selected_tids.extend(chosen)

            except Exception as e:
                logger.error(f"Clustering failed for label {label}: {e}")
                continue

        logger.info(f"Diversity Strategy selected {len(selected_tids)} trajectories.")
        return selected_tids


class SelectTrajectoryBasedOnRepresentativeness(TrajectorySelectionMethod):
    """
    Selects most representative (central) trajectories using Z-scores.
    Parameters: None specific currently.
    """

    def __init__(self, proportion: float, seed: int = 1415, params: Optional[Dict[str, Any]] = None):
        super().__init__(proportion, seed)
        self.params = params or {}

    def select(self, trajectory_feats_df: pd.DataFrame) -> List[str]:
        df = trajectory_feats_df.reset_index() if 'tid' not in trajectory_feats_df.columns else trajectory_feats_df.copy()

        unique_tids = df['tid'].unique()
        unique_labels = [l for l in df['label'].unique() if pd.notna(l)]
        selected_tids = []

        total_to_select = int(len(unique_tids) * self.proportion)
        total_rows = len(df[df['label'].isin(unique_labels)])

        for label in unique_labels:
            label_df = df[df['label'] == label]
            if label_df.empty: continue

            # Proportional allocation
            num_to_select = int(total_to_select * (len(label_df) / total_rows))
            num_to_select = min(num_to_select, len(label_df))
            if num_to_select == 0: continue

            tids = label_df['tid'].values
            feature_cols = [c for c in label_df.columns if c not in ['tid', 'label', 'time']]
            features = label_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

            # Vectorized Z-Score Calculation
            mean_vals = features.mean(axis=0)
            std_vals = features.std(axis=0)
            std_vals[std_vals < 1e-9] = 1.0  # Avoid div/0

            z_scores = (features - mean_vals) / std_vals

            # Representativeness = Sum of Absolute Z-Scores (Lower is more representative/central)
            rep_scores = z_scores.abs().sum(axis=1).values

            # Sort Ascending (Lowest Z-score sum = Closest to mean)
            sorted_indices = np.argsort(rep_scores)
            top_indices = sorted_indices[:num_to_select]

            selected_tids.extend(tids[top_indices])

        logger.info(f"Representativeness Strategy selected {len(selected_tids)} trajectories.")
        return selected_tids


class SelectTrajectoryBasedOnUncertainty(TrajectorySelectionMethod):
    """
    Selects trajectories where a model is most uncertain. This is a form of
    Active Learning, where we augment the data the model finds most confusing.

    This implementation uses a simple, fast 'scout' model to estimate uncertainty.
    Uncertainty is quantified by the entropy of the predicted class probabilities.
    """

    def __init__(self, proportion: float, seed: int = 1415, params: Optional[Dict[str, Any]] = None):
        super().__init__(proportion, seed)
        self.params = params or {}
        # We use a simple, fast, and probabilistic model as our "scout"
        # The pipeline ensures the model can handle multiclass classification
        self.scout_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=self.seed, solver='lbfgs', multi_class='auto', max_iter=1000)
        )

    def select(self, trajectory_feats_df: pd.DataFrame) -> List[str]:
        df = trajectory_feats_df.reset_index() if 'tid' not in trajectory_feats_df.columns else trajectory_feats_df.copy()

        feature_cols = [c for c in df.columns if c not in ['tid', 'label', 'time']]
        features = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        labels = df['label'].values
        tids = df['tid'].values

        if len(np.unique(labels)) < 2:
            logger.warning("Uncertainty selection requires at least 2 classes. Skipping.")
            return []

        logger.info("Uncertainty Strategy: Training scout model to find confusing trajectories...")
        # 1. Train the scout model on the entire training set
        self.scout_model.fit(features, labels)

        # 2. Predict probabilities for each trajectory
        predicted_probabilities = self.scout_model.predict_proba(features)

        # 3. Calculate entropy for each prediction. High entropy = high uncertainty.
        uncertainty_scores = entropy(predicted_probabilities.T)

        # Create a DataFrame to manage scores and TIDs
        scores_df = pd.DataFrame({
            'tid': tids,
            'label': labels,
            'uncertainty': uncertainty_scores
        })

        # 4. Perform stratified selection based on the highest uncertainty
        selected_tids = []
        # Group by label and select the top N most uncertain trajectories from each group
        for label, group in scores_df.groupby('label'):
            group = group.sort_values(by='uncertainty', ascending=False)

            # Calculate how many to select for this label
            num_to_select = int(len(group) * self.proportion)
            num_to_select = max(1, num_to_select)  # Select at least one if proportion is > 0

            selected_tids.extend(group.head(num_to_select)['tid'].tolist())

        logger.info(f"Uncertainty Strategy selected {len(selected_tids)} trajectories.")
        return selected_tids