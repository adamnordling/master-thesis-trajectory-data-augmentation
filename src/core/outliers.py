import logging
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Any, Union

# Initialize logger
logger = logging.getLogger(__name__)


def detect_outliers_dbos(
        dataset: np.ndarray,
        d: float = 1.0,
        fraction: float = 0.05
) -> Dict[str, Any]:
    """
    Implements Density-Based Outlier Selection (DBOS).

    Identifies outliers based on the density of neighbors within a hypersphere
    of radius `d`. Points with fewer neighbors than the threshold are flagged.

    Args:
        dataset: Numeric numpy array of shape (n_samples, n_features).
        d: Radius parameter for the neighborhood search.
        fraction: The expected fraction of outliers in the dataset (0 to 1).

    Returns:
        Dictionary containing:
            - 'neighbors': Count of neighbors for each point.
            - 'scores': Normalized outlier scores (0 to 1, higher is more outlier-ish).
            - 'classification': Array of strings ('Outlier' or 'Inlier').
    """
    # Convert to numpy array if not already
    if not isinstance(dataset, np.ndarray):
        dataset = np.array(dataset)

    # Validate inputs
    if not np.issubdtype(dataset.dtype, np.number):
        logger.error("Dataset input contains non-numeric data.")
        raise ValueError('Dataset input is not numeric')

    if not isinstance(d, (int, float)) or not isinstance(fraction, (int, float)):
        raise ValueError('Parameters d and fraction must be numeric')

    n_samples = dataset.shape[0]
    logger.debug(f"Running DBOS on {n_samples} samples with radius={d:.4f}, fraction={fraction}")

    # Use Ball Tree algorithm for efficient radius queries in higher dimensions
    nn = NearestNeighbors(radius=d, algorithm='ball_tree', metric='euclidean')
    nn.fit(dataset)

    # Get neighbor counts efficiently
    # return_distance=False is faster as we only need the count
    neighbor_indices = nn.radius_neighbors(dataset, return_distance=False)

    # Calculate neighborhood size (subtract 1 to exclude the point itself)
    neighborhood = np.array([len(neighbors) - 1 for neighbors in neighbor_indices])

    # Determine threshold based on fraction
    threshold = n_samples * fraction

    # Classify each observation
    classification = np.where(neighborhood < threshold, 'Outlier', 'Inlier')

    num_outliers = np.sum(classification == 'Outlier')
    logger.info(f"DBOS detection complete. Found {num_outliers} outliers ({num_outliers / n_samples:.1%}).")

    # Invert neighborhood counts for outlier scoring
    # Fewer neighbors = Higher outlier score
    inverted_neighborhood = neighborhood.max() - neighborhood

    # Normalize scores to 0-1 range
    scaler = MinMaxScaler()
    outlier_scores = scaler.fit_transform(inverted_neighborhood.reshape(-1, 1))

    # Return results
    return {
        'neighbors': neighborhood,
        'scores': outlier_scores,
        'classification': classification
    }


def find_average_distance(dataset: np.ndarray, sample_size: int = 1000) -> float:
    """
    Calculates the average pairwise Euclidean distance for a dataset.

    For datasets larger than `sample_size`, it estimates the distance using
    random sampling to avoid O(N^2) memory and time complexity.

    Args:
        dataset: Numeric numpy array.
        sample_size: Max number of samples to use for calculation.

    Returns:
        The average pairwise distance (float).
    """
    if not isinstance(dataset, np.ndarray):
        dataset = np.array(dataset)

    if not np.issubdtype(dataset.dtype, np.number):
        raise ValueError('Dataset input is not numeric')

    # Sampling strategy for scalability
    if len(dataset) > sample_size:
        logger.debug(f"Dataset size ({len(dataset)}) > {sample_size}. Sampling for distance calculation.")
        # Ensure reproducibility if needed by setting seed externally or adding param
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        sample = dataset[indices]
    else:
        sample = dataset

    # Compute pairwise distances on the sample
    # pdist returns a condensed distance matrix (1D array)
    try:
        pairwise_distances = pdist(sample, metric='euclidean')

        # Handle case where sample might be too small (0 or 1 point)
        if len(pairwise_distances) == 0:
            return 0.0

        average_distance = np.mean(pairwise_distances)
        return float(average_distance)

    except Exception as e:
        logger.warning(f"Failed to calculate average distance: {e}. Defaulting to 1.0.")
        return 1.0