import os

import pandas as pd
import yfinance as yf

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_data(
    ticker: str = "AAPL",
    interval: str = "10m",
    period: str = "3mo",
    filename: str = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Load historical stock data from a local cache (CSV) or Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL").
        interval (str): Candle interval.
        period (str): Period for historical data.
        filename (str): Path to the CSV file. If None, generated automatically.
        force_download (bool): If True, always download data instead of using cache.

    Returns:
        pd.DataFrame: DataFrame containing OHLCV data indexed by datetime.
    """
    if filename is None:
        filename = f"{ticker}_{interval}_{period}.csv"

    # 1. Try to load data from local file (if available and not forcing download)
    if not force_download and os.path.exists(filename):
        print(f"Loading data from file: {filename}")
        try:
            data = pd.read_csv(filename, index_col=0, parse_dates=True)
            return data
        except Exception as e:
            print(f"Error reading {filename}: {e}. Downloading new data...")

    # 2. Download data from Yahoo Finance if file is missing or corrupted
    print(f"Downloading data for {ticker} ({interval}, {period})...")
    try:
        data = yf.download(ticker, interval=interval, period=period)

        # If columns are multi-indexed (e.g., 'Adj Close'), flatten them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Keep only OHLCV columns
        data = data[OHLCV_COLUMNS]

        # Save to CSV for caching
        data.to_csv(filename)
        print(f"Data saved to file: {filename}")

        return data

    except Exception as e:
        raise RuntimeError(f"Failed to download data for {ticker}: {e}")


def resample_to_10m(data: pd.DataFrame) -> None:
    return (
        data.resample("10min")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )
