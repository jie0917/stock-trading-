"""Data fetching utilities for US stocks via yfinance."""
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def fetch_prices(tickers, start=None, end=None, years=5, price_field="Close"):
    """Download historical prices for one or more tickers.

    Returns a DataFrame indexed by date with one column per ticker.
    """
    if end is None:
        end = datetime.today()
    if start is None:
        start = end - timedelta(days=365 * years)

    # threads=False: yfinance's download threads can print a harmless
    # "Exception ignored in _DeleteDummyThreadOnDel" at exit on Python 3.13.
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False,
                       threads=False)[price_field]
    if isinstance(data, pd.Series):
        name = tickers if isinstance(tickers, str) else tickers[0]
        data = data.to_frame(name=name)
    return data.dropna(how="all")


def align_to_common_start(prices: pd.DataFrame) -> pd.DataFrame:
    """Truncate every column to the latest common start date (the ticker with
    the shortest history), so all tickers cover the same date range."""
    common_start = max(
        idx for idx in (prices[t].first_valid_index() for t in prices.columns)
        if idx is not None
    )
    return prices.loc[common_start:]
