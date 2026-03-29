import logging
import os
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_dataframe(filepath: str, dtype: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Robustly loads a DataFrame from CSV or Feather format.

    Enforces coordinate precision and Trajectory ID string types upon loading.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # Use Any for values to satisfy Mypy's complex Union requirements for read_csv
    default_dtype: Dict[str, Any] = {"tid": str, "label": str}
    if dtype:
        default_dtype.update(dtype)

    try:
        # 1. Identify and load format
        if filepath.endswith(".feather"):
            # Note: memory_map is handled internally by pyarrow;
            # not exposed as a kwarg in the pandas wrapper.
            df = pd.read_feather(filepath)
        elif filepath.endswith(".csv"):
            df = pd.read_csv(filepath, dtype=default_dtype, engine='pyarrow')
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        # 2. Standardize data (Post-Load)

        # Ensure TID is always a string (prevents leading zero bug)
        if "tid" in df.columns:
            df["tid"] = df["tid"].astype(str)

        # Ensure Time is datetime
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")

        # Ensure Coords are float64 (Scientific precision for geodetic math)
        if {"lat", "lon"}.issubset(df.columns):
            df[["lat", "lon"]] = df[["lat", "lon"]].astype("float64")

        return df

    except Exception as e:
        logger.error(f"Failed to load file {filepath}: {e}")
        raise


def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """
    Saves a DataFrame to CSV or Feather, creating parent directories if needed.

    Utilizes LZ4 compression for Feather files to balance speed and disk usage.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if filepath.endswith(".feather"):
            # reset_index is required for Feather format.
            # compression='lz4' provides high-speed I/O for trajectory data.
            df.reset_index(drop=True).to_feather(filepath, compression="lz4")
        elif filepath.endswith(".csv"):
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported export format: {filepath}")

    except Exception as e:
        logger.error(f"Failed to save file {filepath}: {e}")
        raise
