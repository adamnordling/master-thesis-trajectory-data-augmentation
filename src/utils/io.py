import os
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def load_dataframe(filepath: str, dtype: Optional[dict] = None) -> pd.DataFrame:
    """
    Robustly loads a DataFrame from CSV or Feather format.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    default_dtype = {'tid': str, 'label': str}
    if dtype:
        default_dtype.update(dtype)

    try:
        # 1. IDENTIFY AND LOAD FORMAT
        if filepath.endswith('.feather'):
            # memory_map is for READING only. It makes Optuna trials much faster.
            df = pd.read_feather(filepath, memory_map=True)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath, dtype=default_dtype)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        # 2. STANDARDIZE DATA (Post-Load)

        # Ensure TID is always a string (prevents leading zero bug)
        if 'tid' in df.columns:
            df['tid'] = df['tid'].astype(str)

        # Ensure Time is datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')

        # Ensure Coords are float64 (Scientific precision)
        if {'lat', 'lon'}.issubset(df.columns):
            df[['lat', 'lon']] = df[['lat', 'lon']].astype('float64')

        return df

    except Exception as e:
        logger.error(f"Failed to load file {filepath}: {e}")
        raise


def save_dataframe(df: pd.DataFrame, filepath: str):
    """
    Saves a DataFrame to CSV or Feather, creating parent directories if needed.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if filepath.endswith('.feather'):
            # reset_index is required for Feather format.
            # compression='lz4' is the best for speed/space.
            # NOTE: memory_map is NOT used here.
            df.reset_index(drop=True).to_feather(filepath, compression='lz4')
        elif filepath.endswith('.csv'):
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported export format: {filepath}")

    except Exception as e:
        logger.error(f"Failed to save file {filepath}: {e}")
        raise