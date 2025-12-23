import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Prepares feature matrix (X) and target vector (y) from a dataframe.

    Responsibilities:
    1. Separates Features (X) from Metadata (tid) and Targets (label).
    2. Cleans infinite/NaN values arising from division-by-zero errors.
    3. Encodes string labels (e.g., 'car', 'bus') into integers (0, 1).

    Args:
        df: DataFrame containing 'tid', 'label', and feature columns.

    Returns:
        X: Feature DataFrame (clean).
        y_encoded: Series of integer labels.
        le: The fitted LabelEncoder (useful if we need to inverse transform later).
    """
    if 'label' not in df.columns or 'tid' not in df.columns:
        raise ValueError("Input DataFrame must contain 'tid' and 'label' columns.")

    # Drop metadata to isolate features
    x = df.drop(columns=['tid', 'label'])
    y = df['label']

    # 1. Convert Infs to NaN so they can be calculated
    x = x.replace([np.inf, -np.inf], np.nan)

    # 2. Compute medians for all columns
    column_medians = x.median()

    # 3. Fill NaNs with the median (Fixes most cases)
    x = x.fillna(column_medians)

    # 4. FINAL CATCH-ALL:
    # If a column was 100% empty, the median is NaN.
    # We fill those with 0 so the ML model doesn't crash.
    x = x.fillna(0)

    # Encode target labels
    df = df.dropna(subset=['label'])
    y = df['label'].astype(str)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Return as Series with index preserved to align with X
    y_series = pd.Series(y_encoded, index=y.index, name='label')

    return x, y_series, le