"""Dollar-cost averaging (DCA) simulation: invest a fixed amount every month."""
import pandas as pd

from .data import align_to_common_start


def simulate_dca(prices: pd.DataFrame, monthly_investment: float = 1000.0, align: bool = False):
    """Simulate investing a fixed amount at the start of every calendar month
    into each ticker independently (e.g. "$1,000/month into each stock in the list").

    The buy price each month is the ticker's first available closing price in
    that calendar month (its listed price on that month's `resample("MS")` bucket).

    Parameters
    ----------
    prices : pd.DataFrame
        Historical prices indexed by date, one column per ticker (e.g. from
        `regression_testing.data.fetch_prices`).
    monthly_investment : float
        Amount invested into EACH ticker at the start of every month (default $1,000).
    align : bool
        If True, truncate every ticker to the latest common start date (the
        ticker with the shortest history) before backtesting, so all tickers
        cover the same date range. If False (default), each ticker uses its
        own full available history.

    Returns
    -------
    dict with:
        - "value": pd.DataFrame — portfolio value over time, one column per
          ticker plus a "Total" column (sum across tickers).
        - "invested": pd.DataFrame — cumulative amount contributed over time,
          same columns as "value".
        - "summary": dict keyed by ticker (plus "Total") with final_value,
          total_invested, gain, total_return_pct, num_purchases.
    """
    if align:
        prices = align_to_common_start(prices)

    value = pd.DataFrame(index=prices.index)
    invested = pd.DataFrame(index=prices.index)
    summary = {}

    for ticker in prices.columns:
        s = prices[ticker].dropna()
        if s.empty:
            continue

        # Price on the first trading day of each calendar month (the "buy" price).
        monthly_price = s.resample("MS").first().dropna()
        shares_bought = monthly_investment / monthly_price
        cum_shares = shares_bought.cumsum()
        cum_invested = pd.Series(monthly_investment, index=monthly_price.index).cumsum()

        # Carry cumulative shares/contributions forward across all trading days
        # (0 before the first purchase).
        value[ticker] = cum_shares.reindex(s.index, method="ffill").fillna(0) * s
        invested[ticker] = cum_invested.reindex(s.index, method="ffill").fillna(0)

        final_value = value[ticker].iloc[-1]
        total_invested = invested[ticker].iloc[-1]
        summary[ticker] = {
            "final_value": final_value,
            "total_invested": total_invested,
            "gain": final_value - total_invested,
            "total_return_pct": (final_value / total_invested - 1) * 100 if total_invested else None,
            "num_purchases": len(monthly_price),
        }

    value = value.fillna(0)
    invested = invested.fillna(0)
    value["Total"] = value.sum(axis=1)
    invested["Total"] = invested.sum(axis=1)

    final_total = value["Total"].iloc[-1]
    invested_total = invested["Total"].iloc[-1]
    summary["Total"] = {
        "final_value": final_total,
        "total_invested": invested_total,
        "gain": final_total - invested_total,
        "total_return_pct": (final_total / invested_total - 1) * 100 if invested_total else None,
        "num_purchases": None,
    }

    return {"value": value, "invested": invested, "summary": summary}
