import logging
from typing import Any, Dict, List, Union, cast

import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.stats import kurtosis, skew

# Initialize logger
logger = logging.getLogger(__name__)

# Define the geodetic object once for reuse (Module-level constant)
_GEOD = Geod(ellps="WGS84")


def _calculate_distances_vectorized(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> List[float]:
    """
    Fast and accurate vectorized distance calculation using pyproj.

    Returns distances in meters.
    """
    if len(lat1) == 0:
        return []

    _, _, distances_meters = _GEOD.inv(lon1, lat1, lon2, lat2)
    # Cast to List[float] to satisfy Mypy's strict return type check
    return cast(List[float], distances_meters.tolist())


def _compute_stats(data_in: Union[np.ndarray, List[float]], prefix: str) -> Dict[str, float]:
    """
    Compute statistical features (mean, std, skew, kurt, quantiles).

    Calculates 19 standard kinematic features for a given distribution.
    """
    # Force input into a numpy array so .size and math operations work consistently
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

    # Handle empty data or all-NaNs
    if data.size == 0:
        return {f"{prefix}_{stat}": 0.0 for stat in stat_keys}

    data = data[~np.isnan(data)]
    if data.size == 0:
        return {f"{prefix}_{stat}": 0.0 for stat in stat_keys}

    # --- CALCULATIONS ---
    data_range = np.ptp(data)
    std_val = np.std(data)
    mean_val = np.mean(data)

    # Handle constant values (zero variance) using robust float comparison
    if np.isclose(data_range, 0, atol=1e-9) or np.isclose(std_val, 0, atol=1e-9):
        return {
            f"{prefix}_0s": float(np.sum(data == 0)),
            f"{prefix}_mean": float(mean_val),
            f"{prefix}_meanse": 0.0,
            f"{prefix}_quant_min": float(np.min(data)),
            f"{prefix}_quant_05": float(np.percentile(data, 5)),
            f"{prefix}_quant_10": float(np.percentile(data, 10)),
            f"{prefix}_quant_25": float(np.percentile(data, 25)),
            f"{prefix}_quant_median": float(np.median(data)),
            f"{prefix}_quant_75": float(np.percentile(data, 75)),
            f"{prefix}_quant_90": float(np.percentile(data, 90)),
            f"{prefix}_quant_95": float(np.percentile(data, 95)),
            f"{prefix}_quant_max": float(np.max(data)),
            f"{prefix}_range": float(data_range),
            f"{prefix}_sd": float(std_val),
            f"{prefix}_vcoef": 0.0,
            f"{prefix}_mad": 0.0,
            f"{prefix}_iqr": 0.0,
            f"{prefix}_skew": 0.0,
            f"{prefix}_kurt": 0.0,
        }

    # Complex stats
    vcoef_val = std_val / mean_val if abs(mean_val) > 1e-9 else 0.0
    q75, q25 = np.percentile(data, [75, 25])
    iqr_val = q75 - q25

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
        f"{prefix}_vcoef": float(vcoef_val),
        f"{prefix}_mad": float(np.median(np.abs(data - np.median(data)))),
        f"{prefix}_iqr": float(iqr_val),
        f"{prefix}_skew": float(skew(data)),
        f"{prefix}_kurt": float(kurtosis(data)),
    }


def extract_trajectory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts geometric and kinematic features from trajectory data.

    Processes trajectory points grouped by TID to generate fractal and kinematic descriptors.
    """
    features: List[Dict[str, Any]] = []

    required_cols = {"tid", "lat", "lon", "time", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input DataFrame missing columns: {required_cols - set(df.columns)}")

    groups = df.groupby("tid")

    for tid, group in groups:
        group = group.sort_values("time").reset_index(drop=True)
        trajectory_features: Dict[str, Any] = {"tid": str(tid)}

        if len(group) > 1:
            lats = group["lat"].to_numpy(dtype="float64")
            lons = group["lon"].to_numpy(dtype="float64")
            segment_lengths = _calculate_distances_vectorized(lats[:-1], lons[:-1], lats[1:], lons[1:])

            # Mypy suppression for known Pandas-stubs Timedelta limitation
            time_diffs = group["time"].diff().dt.total_seconds().to_numpy()[1:]  # type: ignore[attr-defined]
        else:
            segment_lengths = []
            time_diffs = np.array([], dtype=float)

        # --- 1. Distance-based geometry (Fractal Dimensions) ---
        start_coords_lat = []
        start_coords_lon = []
        end_coords_lat = []
        end_coords_lon = []
        path_distances = []
        is_valid_segment = []

        for level in range(1, 6):
            num_segments = level
            segment_size = len(group) // num_segments

            for j in range(num_segments):
                start = j * segment_size
                end = (j + 1) * segment_size if (j + 1) * segment_size < len(group) else len(group) - 1

                if start < end:
                    start_coords_lat.append(group.iloc[start]["lat"])
                    start_coords_lon.append(group.iloc[start]["lon"])
                    end_coords_lat.append(group.iloc[end]["lat"])
                    end_coords_lon.append(group.iloc[end]["lon"])

                    total_path_distance = sum(segment_lengths[start:end]) if segment_lengths else 0.0
                    path_distances.append(total_path_distance)
                    is_valid_segment.append(True)
                else:
                    is_valid_segment.append(False)

        signatures = []
        if any(is_valid_segment):
            try:
                _, _, segment_distances_meters = _GEOD.inv(
                    start_coords_lon, start_coords_lat, end_coords_lon, end_coords_lat
                )
                denom = np.maximum(np.array(path_distances), 0.001)
                calculated_signatures = segment_distances_meters / denom
            except Exception:
                calculated_signatures = np.zeros(len(path_distances))

            sig_idx = 0
            for is_valid in is_valid_segment:
                if is_valid:
                    signatures.append(calculated_signatures[sig_idx])
                    sig_idx += 1
                else:
                    signatures.append(0.0)
        else:
            signatures = [0.0] * (15)  # Sum of segments for levels 1-5 is 15

        # Assign columns
        idx = 0
        for level in range(1, 6):
            for segment in range(1, level + 1):
                col_name = f"distance_geometry_{level}_{segment}"
                trajectory_features[col_name] = signatures[idx] if idx < len(signatures) else 0.0
                idx += 1

        # --- 2. Kinematics ---
        speeds = np.array([], dtype=float)
        angles = np.array([], dtype=float)

        if len(group) > 1:
            seg_len_arr = np.array(segment_lengths)
            speeds = np.divide(
                seg_len_arr, time_diffs, out=np.zeros_like(seg_len_arr, dtype=float), where=time_diffs != 0
            )

            if len(group) > 2:
                # v = np.diff(group[["lat", "lon"]].to_numpy(), axis=0) # Original line
                # Ensuring high precision for vector math
                coords = group[["lat", "lon"]].to_numpy(dtype="float64")
                v = np.diff(coords, axis=0)
                v1, v2 = v[:-1], v[1:]
                dot = np.einsum("ij,ij->i", v1, v2)
                norm_v1 = np.linalg.norm(v1, axis=1)
                norm_v2 = np.linalg.norm(v2, axis=1)

                valid = (norm_v1 > 0) & (norm_v2 > 0)
                cos_theta = np.ones(len(valid))
                cos_theta[valid] = dot[valid] / (norm_v1[valid] * norm_v2[valid])
                angles = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        accelerations = np.array([], dtype=float)
        if len(speeds) > 1:
            speed_diffs = np.diff(speeds)
            accel_times = (time_diffs[:-1] + time_diffs[1:]) / 2
            accelerations = np.divide(
                speed_diffs, accel_times, out=np.zeros_like(speed_diffs, dtype=float), where=accel_times != 0
            )

        trajectory_features.update(_compute_stats(speeds, "speed"))
        trajectory_features.update(_compute_stats(accelerations, "acceleration"))
        trajectory_features.update(_compute_stats(angles, "angles"))

        trajectory_features["label"] = group["label"].iloc[0]
        features.append(trajectory_features)

    df_out = pd.DataFrame(features)
    logger.info(f"Feature extraction: Processed {len(df_out)} trajectories")
    return df_out
