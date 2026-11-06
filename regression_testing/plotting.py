"""Plotting helpers built on pandas' built-in (matplotlib-backed) .plot()."""
import matplotlib.pyplot as plt
import pandas as pd


def plot_price_trend(prices: pd.Series, fitted: pd.Series, ticker: str, ax=None, show=True):
    df = pd.DataFrame({ticker: prices, "Trend (OLS)": fitted})
    ax = df.plot(ax=ax, figsize=(12, 6), title=f"{ticker} Price vs. Linear Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_return_scatter(y: pd.Series, x: pd.Series, fitted: pd.Series, ticker_y: str, ticker_x: str, ax=None, show=True):
    df = pd.concat([x, y], axis=1)
    df.columns = [ticker_x, ticker_y]
    ax = df.plot.scatter(x=ticker_x, y=ticker_y, ax=ax, figsize=(8, 8),
                          title=f"{ticker_y} vs. {ticker_x} Daily Returns")
    order = df[ticker_x].sort_values().index
    ax.plot(df.loc[order, ticker_x], fitted.loc[order], color="red", label="OLS fit")
    ax.legend()
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_cumulative_returns(prices: pd.DataFrame, ax=None, show=True):
    # Compute each ticker's cumulative return independently so tickers with a
    # shorter history (e.g. newer ETFs) start from their own first available date
    # instead of being truncated to the latest common start date.
    cum = prices.apply(lambda col: (1 + col.pct_change().dropna()).cumprod())
    ax = cum.plot(ax=ax, figsize=(12, 6), title="Cumulative Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_calendar_year_returns(year_returns: pd.DataFrame, ax=None, show=True):
    """Line chart of each ticker's return (%) per calendar year, from
    `regression_testing.regression.calendar_year_returns`."""
    ax = year_returns.plot(ax=ax, marker="o", figsize=(12, 6), title="Return by Calendar Year")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Return (%)")
    ax.legend(title="Ticker", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_dca_growth(value: pd.DataFrame, invested: pd.DataFrame = None, ax=None, show=True):
    """Plot dollar-cost-averaging portfolio value over time, one line per ticker.

    `value` and `invested` are the DataFrames returned by
    `regression_testing.dca.simulate_dca` (a "Total" column, if present, is
    excluded from the per-ticker lines). If `invested` is given, the average
    per-ticker contribution is overlaid as a dashed reference line so growth
    is easy to compare against money put in.
    """
    cols = [c for c in value.columns if c != "Total"]
    ax = value[cols].plot(ax=ax, figsize=(12, 6), title="DCA Value: $1,000/month per ticker")
    if invested is not None:
        ref_cols = [c for c in cols if c in invested.columns]
        ref = invested[ref_cols].mean(axis=1)
        ax.plot(ref.index, ref.values, linestyle="--", color="black", alpha=0.6, label="Amount invested/ticker")
        ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Value ($)")
    plt.tight_layout()
    if show:
        plt.show()
    return ax
