from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scale_ohlcv(
    data: pd.DataFrame, log_volume: bool = True
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Scale OHLCV data using MinMaxScaler.

    - Open, High, Low are scaled together in [0, 1].
    - Close is scaled separately (for inverse transformation later).
    - Volume is log-transformed (optional) and scaled in [0, 1].

    Args:
        data (pd.DataFrame): Input OHLCV data.
        log_volume (bool): Whether to apply log1p to volume before scaling.

    Returns:
        Tuple[pd.DataFrame, MinMaxScaler]:
            - Scaled DataFrame with the same index as input.
            - "Close" column scaler.
    """
    # Split columns
    ohl_cols = ["Open", "High", "Low"]
    close_col = ["Close"]
    vol_col = ["Volume"]

    # Extract subsets
    ohl = data[ohl_cols].copy()
    close = data[close_col].copy()
    volume = data[vol_col].copy()

    # Define scalers
    scaler_ohl = MinMaxScaler()
    scaler_close = MinMaxScaler()
    scaler_volume = MinMaxScaler()

    # Fit-transform subsets
    ohl_scaled = scaler_ohl.fit_transform(ohl)
    close_scaled = scaler_close.fit_transform(close)
    if log_volume:
        vol_scaled = scaler_volume.fit_transform(np.log1p(volume))
    else:
        vol_scaled = scaler_volume.fit_transform(volume)

    # Concatenate back into one DataFrame
    scaled_df = pd.DataFrame(
        np.hstack([ohl_scaled, close_scaled, vol_scaled]),
        columns=ohl_cols + close_col + vol_col,
        index=data.index,
    )

    return scaled_df, scaler_close


def create_sequences(
    data: pd.DataFrame, seq_length: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert time series into supervised learning sequences.

    Args:
        data (pd.DataFrame): Input scaled OHLCV data.
        seq_length (int): Number of time steps in each input sequence.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X: array of shape (n_samples, seq_length, n_features)
            - y: array of shape (n_samples,)
    """
    X, y = [], []

    for i in range(len(data) - seq_length):
        # Input sequence: past `seq_length` timesteps
        X.append(data.iloc[i : i + seq_length].values)

        # Target: 'Close' price at the next timestep
        y.append(data.iloc[i + seq_length]["Close"])

    return np.array(X), np.array(y)


def train_val_test_split(
    data: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series into train/validation/test sets.

    Args:
        data (pd.DataFrame): Input time series.
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - train set
            - validation set
            - test set
    """
    n = len(data)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = data.iloc[:train_end]
    val = data.iloc[train_end:val_end]
    test = data.iloc[val_end:]

    return train, val, test
