import logging
from typing import Any, cast

import numpy as np
import pandas as pd
from numba import njit
from pyproj import Geod
from scipy.stats import kurtosis, skew

# Initialize logger
logger = logging.getLogger(__name__)

# Define the geodetic object once for reuse
_GEOD = Geod(ellps="WGS84")


@njit(cache=True)
def _compute_fractal_path_sums_jit(segment_lengths: np.ndarray, group_size: int) -> np.ndarray:
    """Computes path sums for fractal levels 1-5 at C-speed using Numba."""
    path_sums = np.zeros(15, dtype=np.float64)
    idx = 0
    for level in range(1, 6):
        segment_size = group_size // level
        for j in range(level):
            start = j * segment_size
            end = (j + 1) * segment_size
            if end >= group_size:
                end = group_size - 1

            if start < end:
                path_sums[idx] = np.sum(segment_lengths[start:end])
            idx += 1
    return path_sums


@njit(cache=True)
def _get_fractal_indices_jit(group_size: int) -> np.ndarray:
    """Calculates start/end indices for all 15 fractal segments."""
    indices = np.zeros((15, 2), dtype=np.int64)
    idx = 0
    for level in range(1, 6):
        segment_size = group_size // level
        for j in range(level):
            start = j * segment_size
            end = (j + 1) * segment_size
            if end >= group_size:
                end = group_size - 1
            indices[idx, 0] = start
            indices[idx, 1] = end
            idx += 1
    return indices


def _calculate_distances_vectorized(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> list[float]:
    """Fast and accurate vectorized distance calculation using pyproj."""
    if len(lat1) == 0:
        return []
    _, _, distances_meters = _GEOD.inv(lon1, lat1, lon2, lat2)
    return cast(list[float], distances_meters.tolist())


def _compute_stats(data_in: np.ndarray | list[float], prefix: str) -> dict[str, float]:
    """Compute statistical features (mean, std, skew, kurt, quantiles)."""
    data: np.ndarray = np.asarray(data_in)
    stat_keys = [
        "0s",
        "mean",
        "meanse",
        "quant_min",
        "quant_05",
        "quant_10",
        "quant_25",
        "quant_median",
        "quant_75",
        "quant_90",
        "quant_95",
        "quant_max",
        "range",
        "sd",
        "vcoef",
        "mad",
        "iqr",
        "skew",
        "kurt",
    ]

    if data.size == 0:
        return {f"{prefix}_{stat}": 0.0 for stat in stat_keys}

    data = data[~np.isnan(data)]
    if data.size == 0:
        return {f"{prefix}_{stat}": 0.0 for stat in stat_keys}

    data_range = np.ptp(data)
    std_val = np.std(data)
    mean_val = np.mean(data)

    if np.isclose(data_range, 0, atol=1e-9) or np.isclose(std_val, 0, atol=1e-9):
        res = {f"{prefix}_{stat}": 0.0 for stat in stat_keys}
        res.update(
            {
                f"{prefix}_mean": float(mean_val),
                f"{prefix}_quant_min": float(np.min(data)),
                f"{prefix}_quant_max": float(np.max(data)),
                f"{prefix}_range": float(data_range),
                f"{prefix}_sd": float(std_val),
            }
        )
        return res

    q75, q25 = np.percentile(data, [75, 25])

    return {
        f"{prefix}_0s": float(np.sum(data == 0)),
        f"{prefix}_mean": float(mean_val),
        f"{prefix}_meanse": float(std_val / np.sqrt(data.size)),
        f"{prefix}_quant_min": float(np.min(data)),
        f"{prefix}_quant_05": float(np.percentile(data, 5)),
        f"{prefix}_quant_10": float(np.percentile(data, 10)),
        f"{prefix}_quant_25": float(q25),
        f"{prefix}_quant_median": float(np.median(data)),
        f"{prefix}_quant_75": float(q75),
        f"{prefix}_quant_90": float(np.percentile(data, 90)),
        f"{prefix}_quant_95": float(np.percentile(data, 95)),
        f"{prefix}_quant_max": float(np.max(data)),
        f"{prefix}_range": float(data_range),
        f"{prefix}_sd": float(std_val),
        f"{prefix}_vcoef": float(std_val / mean_val if abs(mean_val) > 1e-9 else 0.0),
        f"{prefix}_mad": float(np.median(np.abs(data - np.median(data)))),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_skew": float(skew(data)),
        f"{prefix}_kurt": float(kurtosis(data)),
    }


def extract_trajectory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts geometric and kinematic features from trajectory data."""
    features: list[dict[str, Any]] = []

    required_cols = {"tid", "lat", "lon", "time", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

    for tid, group in df.groupby("tid"):
        group_sorted = group.sort_values("time").reset_index(drop=True)
        trajectory_features: dict[str, Any] = {"tid": str(tid)}
        group_size = len(group_sorted)

        if group_size > 1:
            lats = group_sorted["lat"].to_numpy(dtype="float64")
            lons = group_sorted["lon"].to_numpy(dtype="float64")
            times = group_sorted["time"].to_numpy()  # NumPy Datetime array

            # 1. Faster Distance Calculation
            segment_lengths = np.array(_calculate_distances_vectorized(lats[:-1], lons[:-1], lats[1:], lons[1:]))

            # 2. Optimized "Deltatime": Pure NumPy subtraction / seconds
            time_diffs = np.diff(times) / np.timedelta64(1, "s")
        else:
            segment_lengths = np.array([], dtype=float)
            time_diffs = np.array([], dtype=float)

        # Fractal Dimensions Logic
        path_distances = _compute_fractal_path_sums_jit(segment_lengths, group_size)
        indices = _get_fractal_indices_jit(group_size)

        s_lats, s_lons = lats[indices[:, 0]], lons[indices[:, 0]]
        e_lats, e_lons = lats[indices[:, 1]], lons[indices[:, 1]]

        signatures = np.zeros(15, dtype=np.float64)
        try:
            _, _, straight_distances = _GEOD.inv(s_lons, s_lats, e_lons, e_lats)
            denom = np.maximum(path_distances, 0.001)
            signatures = straight_distances / denom
        except Exception:
            pass

        sig_idx = 0
        for level in range(1, 6):
            for segment in range(1, level + 1):
                trajectory_features[f"distance_geometry_{level}_{segment}"] = float(signatures[sig_idx])
                sig_idx += 1

        speeds = np.array([], dtype=float)
        angles = np.array([], dtype=float)

        if group_size > 1:
            speeds = np.divide(segment_lengths, time_diffs, out=np.zeros_like(segment_lengths), where=time_diffs != 0)

            if group_size > 2:
                # Vectorized Angles
                v = np.diff(np.column_stack((lats, lons)), axis=0)
                v1, v2 = v[:-1], v[1:]
                dot = np.einsum("ij,ij->i", v1, v2)
                n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
                valid = (n1 > 0) & (n2 > 0)
                cos_theta = np.ones(len(valid))
                cos_theta[valid] = dot[valid] / (n1[valid] * n2[valid])
                angles = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        accelerations = np.array([], dtype=float)
        if len(speeds) > 1:
            accel_times = (time_diffs[:-1] + time_diffs[1:]) / 2
            accelerations = np.divide(
                np.diff(speeds), accel_times, out=np.zeros(len(speeds) - 1), where=accel_times != 0
            )

        trajectory_features.update(_compute_stats(speeds, "speed"))
        trajectory_features.update(_compute_stats(accelerations, "acceleration"))
        trajectory_features.update(_compute_stats(angles, "angles"))
        trajectory_features["label"] = group_sorted["label"].iloc[0]
        features.append(trajectory_features)

    df_out = pd.DataFrame(features)
    logger.info(f"Feature extraction: Processed {len(df_out)} trajectories")
    return df_out
