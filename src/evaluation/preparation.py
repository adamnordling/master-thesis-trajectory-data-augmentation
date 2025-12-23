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

    # Robust Cleaning:
    # 1. Fill NaNs with 0 (often happens in speed calc if time_diff is 0)
    # 2. Replace Infinity with 0
    x = x.fillna(0).replace([np.inf, -np.inf], 0)

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))

    # Return as Series with index preserved to align with X
    y_series = pd.Series(y_encoded, index=y.index, name='label')

    return x, y_series, le