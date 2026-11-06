"""Print today's trend score (the 7 signals behind trend_score) for QQQ or other tickers.

Usage:
    python current_trend_score.py            # QQQ
    python current_trend_score.py QQQ SPY    # several tickers
"""
import sys

import numpy as np

from etf_rotation_strategy import MOMENTUM_MONTHS, SMA_WINDOWS, VOL_LOOKBACK, trend_score
from regression_testing.data import fetch_prices

MIN_SCORE = 3 / 7  # trend guard threshold used by the main strategy
VOL_TARGET = 0.25  # vol target used by the main strategy


def report(ticker: str, price):
    price = price.dropna()
    last = price.iloc[-1]
    date = price.index[-1].date()
    print(f"\n=== {ticker}  {date}  close ${last:,.2f} ===")

    positive = 0
    for n in SMA_WINDOWS:
        sma = price.rolling(n).mean().iloc[-1]
        ok = last > sma
        positive += ok
        print(f"  {'✓' if ok else '✗'} above {n:>3}-day SMA  ${sma:>9,.2f}  ({last / sma - 1:+.1%})")
    for m in MOMENTUM_MONTHS:
        past = price.shift(21 * m).iloc[-1]
        ok = last > past
        positive += ok
        print(f"  {'✓' if ok else '✗'} above {m:>2} months ago  ${past:>9,.2f}  ({last / past - 1:+.1%})")

    score = trend_score(price).iloc[-1]
    total = len(SMA_WINDOWS) + len(MOMENTUM_MONTHS)
    status = "intact" if score >= MIN_SCORE else "BROKEN"
    print(f"\n  Trend score: {positive}/{total} = {score:.2f}  "
          f"(guard needs >= {round(MIN_SCORE * total)}/{total}) -> trend {status}")

    peak = price.rolling(252, min_periods=1).max().iloc[-1]  # 52-week closing high
    vol = price.pct_change().rolling(VOL_LOOKBACK).std().iloc[-1] * np.sqrt(252)
    print(f"  Drawdown from 52-week high ${peak:,.2f}: {last / peak - 1:+.1%}")
    print(f"  {VOL_LOOKBACK}-day realized vol: {vol:.1%}  "
          f"(vol target {VOL_TARGET:.0%} caps 1x exposure at {min(1.0, VOL_TARGET / vol):.0%})")


if __name__ == "__main__":
    tickers = [t.upper() for t in sys.argv[1:]] or ["QQQ"]
    # 12-month momentum and the 200-day SMA need about 13 months of history.
    prices = fetch_prices(tickers, years=2)
    for t in tickers:
        report(t, prices[t])
