import numpy as np
import pandas as pd


def validate_data(data: pd.DataFrame) -> bool:
    """
    Validate raw OHLCV data for common issues.

    Checks:
        - Missing values
        - Non-positive prices (Open, High, Low, Close)
        - Negative volume

    Args:
        data (pd.DataFrame): DataFrame containing OHLCV data.

    Returns:
        bool: True if all checks pass, False otherwise.
    """
    valid = True

    # Check for missing values
    if data.isnull().values.any():
        print("Validation failed: missing values detected")
        print(data.isnull().sum())
        valid = False
    else:
        print("Validation passed: no missing values")

    # Check for non-positive prices
    price_cols = ["Open", "High", "Low", "Close"]
    if (data[price_cols] <= 0).any().any():
        print("Validation failed: non-positive prices detected")
        print(data[data[price_cols] <= 0])
        valid = False
    else:
        print("Validation passed: all prices are positive")

    # Check for negative volume
    if (data["Volume"] < 0).any():
        print("Validation failed: negative volume detected")
        print(data[data["Volume"] < 0])
        valid = False
    else:
        print("Validation passed: all volumes are non-negative")

    return valid


def validate_scaled_data(scaled_data: pd.DataFrame) -> bool:
    """
    Validate scaled OHLCV data for anomalies.

    Checks:
        - NaN values
        - Infinite values
        - Values outside [0, 1] range

    Args:
        scaled_data (pd.DataFrame): Scaled OHLCV data.

    Returns:
        bool: True if all checks pass, False otherwise.
    """
    valid = True

    # Check for NaN values
    if scaled_data.isnull().values.any():
        print("Validation failed: NaN values detected in scaled data")
        valid = False

    # Check for infinite values
    if np.isinf(scaled_data.values).any():
        print("Validation failed: infinite values detected in scaled data")
        valid = False

    # Check range [0, 1]
    if (scaled_data < 0).any().any() or (scaled_data > 1).any().any():
        print("Validation failed: values out of [0, 1] range detected")
        outliers = scaled_data[
            (scaled_data < 0).any(axis=1) | (scaled_data > 1).any(axis=1)
        ]
        print(outliers)
        valid = False
    else:
        print("Validation passed: all values are within [0, 1]")

    return valid
