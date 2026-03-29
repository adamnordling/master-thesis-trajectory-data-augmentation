import logging
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Geod

# Initialize a logger for this specific module
logger = logging.getLogger(__name__)


def _generate_random_points_on_circle_border(
    lats: np.ndarray, lons: np.ndarray, radii: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random points on the border of a circle using robust geodesic calculations.

    Args:
        lats: Array of latitude coordinates.
        lons: Array of longitude coordinates.
        radii: Array of radii in kilometers.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (new_lats, new_lons).
    """
    rng = np.random.default_rng(seed)

    # Define the geodetic object once (using the standard WGS84 ellipsoid)
    geod = Geod(ellps="WGS84")

    # Generate all random angles (bearings) at once
    bearings = rng.uniform(0, 360, size=len(lats))

    # Calculate destination points.
    # geod.fwd requires distances in meters (convert km -> m)
    # Geodetic calculation
    new_lons, new_lats, _ = geod.fwd(lons, lats, bearings, radii * 1000)

    new_lats = np.clip(new_lats, -90.0, 90.0)
    new_lons = (new_lons + 180) % 360 - 180  # Wrap longitudes to [-180, 180]
    return new_lats, new_lons


def _generate_new_geo_points(coords: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate new geographical points based on the input coordinates using midpoint displacement.

    Args:
        coords: Numpy array of shape (N, 4) containing [lat1, lon1, lat2, lon2].
        seed: Random seed.

    Returns:
        Tuple of (new_points_array, radii_km).
    """
    if len(coords) == 0:
        return np.array([]), np.array([])

    lat1, lon1 = coords[:, 0], coords[:, 1]
    lat2, lon2 = coords[:, 2], coords[:, 3]

    # Define the geodetic object once
    geod = Geod(ellps="WGS84")

    # Calculate distances (inv returns: forward_az, back_az, distance_m)
    # We select [2] for distance in meters
    distances_meters = geod.inv(lon1, lat1, lon2, lat2)[2]

    # Convert distances to radii in km (half the distance)
    radii_km = (distances_meters / 1000) / 2

    # Handle NaNs (e.g., identical points resulting in 0 distance)
    radii_km = np.nan_to_num(radii_km, nan=0.0)

    # Generate new points
    new_lats, new_lons = _generate_random_points_on_circle_border(lat1, lon1, radii_km, seed)

    new_points = np.column_stack((new_lats, new_lons))

    return new_points, radii_km


def shift_points_randomly(
    trajectory_df: pd.DataFrame, n_aug_trajs: int, points_proportion: float, seed: int
) -> pd.DataFrame:
    """Augments trajectories by modifying a proportion of their interior points.

    This function takes a DataFrame of trajectories, identifies interior segments,
    and geometrically shifts points based on the distance to their neighbors.

    Args:
        trajectory_df: DataFrame containing ['tid', 'lat', 'lon', 'time', ...].
        n_aug_trajs: Number of augmented versions to generate per trajectory.
        points_proportion: Percentage (0.0 - 1.0) of points to modify per trajectory.
        seed: Base random seed.

    Returns:
        A new DataFrame containing BOTH original and augmented trajectories.
    """
    all_records: list[dict[str, Any]] = []
    unique_tids = trajectory_df["tid"].unique()

    logger.info(f"Starting geometric augmentation for {len(unique_tids)} trajectories.")

    for i, tid in enumerate(unique_tids):
        # Extract single trajectory
        traj = trajectory_df[trajectory_df["tid"] == tid].copy()
        traj = traj.sort_values(by="time").reset_index(drop=True)

        # 1. Archive Original Trajectory
        for _, row in traj.iterrows():
            record = row.to_dict()
            record["augmented"] = 0
            all_records.append(record)

        # Skip short trajectories (need at least Start, End, and 1 interior point)
        if len(traj) < 3:
            continue

        # 2. Generate Augmented Versions
        for aug in range(1, n_aug_trajs + 1):
            try:
                # Deterministic seed generation per trajectory/augmentation pair
                tid_numeric = int(hash(str(tid)) % 1_000_000)
                current_seed = seed + aug + tid_numeric
                rng = np.random.default_rng(current_seed)

                # Identify interior points (excluding start/end)
                available_indices = traj.index[1:-1]
                if len(available_indices) == 0:
                    continue

                # Determine how many points to shift
                num_points_to_select = max(1, int(len(available_indices) * points_proportion))
                num_points_to_select = min(num_points_to_select, len(available_indices))

                # Select indices to modify
                selected_indices = rng.choice(available_indices, size=num_points_to_select, replace=False)

                # Prepare coordinates for vectorized calculation
                # P1: Selected points, P2: The immediate next points
                p1_coords = traj.loc[selected_indices, ["lat", "lon"]].values
                p2_coords = traj.loc[selected_indices + 1, ["lat", "lon"]].values

                selected_points_coords = np.hstack([p1_coords, p2_coords])

                if len(selected_points_coords) == 0:
                    continue

                # Execute geometric shift
                new_points, _ = _generate_new_geo_points(selected_points_coords, current_seed)

                # Construct new trajectory
                new_traj = traj.copy()
                new_traj.loc[selected_indices, "lat"] = new_points[:, 0]
                new_traj.loc[selected_indices, "lon"] = new_points[:, 1]

                # Append to records
                for _, row in new_traj.iterrows():
                    record = row.to_dict()
                    record["augmented"] = aug
                    all_records.append(record)

            except Exception as e:
                logger.error(f"Failed to augment trajectory {tid} (ver: {aug}): {e}", exc_info=True)
                continue

    logger.info(f"Augmentation complete. Created {len(all_records)} total records.")
    final_df = pd.DataFrame.from_records(all_records)
    return final_df
