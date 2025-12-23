import os
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def load_dataframe(filepath: str, dtype: Optional[dict] = None) -> pd.DataFrame:
    """
    Robustly loads a DataFrame from CSV or Feather format.

    Features:
    1. Auto-detects format based on extension.
    2. Enforces 'tid' column to be string (critical for trajectory data).
    3. parses 'time' column to datetime objects automatically.

    Args:
        filepath: Path to the file.
        dtype: Optional dictionary for specific column types.

    Returns:
        pd.DataFrame
    """
    if not os.path.exists(filepath):
        # Professional tip: Raise specific errors so callers can handle them
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # Default types for trajectory data
    # We ALWAYS want IDs to be strings to preserve leading zeros (e.g. "0123")
    default_dtype = {'tid': str, 'label': str}
    if dtype:
        default_dtype.update(dtype)

    try:
        if filepath.endswith('.feather'):
            # Feather preserves types metadata, so dtype arg is less critical but good for safety
            df = pd.read_feather(filepath)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath, dtype=default_dtype)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        # Post-load standardization
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')

        if 'tid' in df.columns:
            df['tid'] = df['tid'].astype(str)

        return df

    except Exception as e:
        logger.error(f"Failed to load file {filepath}: {e}")
        raise


def save_dataframe(df: pd.DataFrame, filepath: str):
    """
    Saves a DataFrame to CSV or Feather, creating parent directories if needed.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if filepath.endswith('.feather'):
            df.reset_index(drop=True).to_feather(filepath)
        elif filepath.endswith('.csv'):
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported export format: {filepath}")

    except Exception as e:
        logger.error(f"Failed to save file {filepath}: {e}")
        raise