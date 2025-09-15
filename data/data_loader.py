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


def resample_to_10m(data: pd.DataFrame) -> pd.DataFrame:
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


def csv_transform(filename: str, resample_10min: bool = True) -> None:
    """
    Load OHLCV stock data from CSV, filter from 2018 onwards,
    optionally resample to 10-minute candles, and save to a new CSV.

    Parameters:
    -----------
    filename : str
        Path to the input CSV file with 'Datetime' and OHLCV columns.
    resample_10min : bool, default=True
        If True, resample to 10-minute candles; otherwise keep original frequency.

    Returns:
    --------
    None
        Saves processed data to CSV.
    """
    print(f"Loading data from {filename}...")
    data = pd.read_csv(filename, parse_dates=["Datetime"])

    print("Filtering data from 2018-01-01 and selecting columns...")
    filtered = data.loc[
        data["Datetime"] >= "2018-01-01",
        ["Datetime", "Open", "High", "Low", "Close", "Volume"],
    ]
    print(f"Filtering done. Rows after filter: {len(filtered)}")

    if resample_10min:
        print("Resampling data to 10-minute candles...")
        filtered.set_index("Datetime", inplace=True)

        filtered = resample_to_10m(filtered)

        filtered = filtered.reset_index()
        print("Resampling done.")

        filtered.to_csv("AAPL_10min_2018_2024.csv", index=False)
        print("10-minute data saved to AAPL_2018_2024_10min.csv")
    else:
        filtered.to_csv("AAPL_2018_2024_1min.csv", index=False)
        print("1-minute data saved to AAPL_2018_2024_1min.csv")
