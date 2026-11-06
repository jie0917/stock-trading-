"""Regression utilities: linear trend regression and beta (vs. benchmark) regression."""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def trend_regression(prices: pd.Series):
    """Fit price ~ time (linear trend) via OLS.

    Returns (model, fitted) where fitted is a Series of predicted prices
    aligned to the input index.
    """
    y = prices.dropna()
    t = np.arange(len(y))
    x = sm.add_constant(t)
    model = sm.OLS(y.values, x).fit()
    fitted = pd.Series(model.predict(x), index=y.index, name="trend_fit")
    return model, fitted


def beta_regression(y: pd.Series, x: pd.Series):
    """Regress series `y` on series `x` (e.g. stock returns vs. benchmark returns).

    Returns (model, fitted) where model.params are ["const", "x"].
    """
    df = pd.concat([y, x], axis=1).dropna()
    df.columns = ["y", "x"]
    x_with_const = sm.add_constant(df["x"])
    model = sm.OLS(df["y"], x_with_const).fit()
    fitted = pd.Series(model.predict(x_with_const), index=df.index, name="fit")
    return model, fitted


def returns_beta(prices: pd.DataFrame, ticker: str, benchmark: str):
    """Compute beta/alpha of `ticker` daily returns vs. `benchmark` daily returns."""
    rets = prices[[ticker, benchmark]].pct_change().dropna()
    return beta_regression(rets[ticker], rets[benchmark])


def calculate_performance_metrics(prices: pd.Series, risk_free_rate: float = 0.02):
    """Calculate common stock performance metrics.

    Parameters:
    -----------
    prices : pd.Series
        Historical price series indexed by date
    risk_free_rate : float
        Annual risk-free rate for Sharpe ratio (default 2%)

    Returns:
    --------
    dict with metrics:
        - total_return: (end - start) / start
        - annual_return: CAGR
        - daily_volatility: std dev of daily returns
        - annual_volatility: annualized volatility
        - sharpe_ratio: (annual_return - risk_free_rate) / annual_volatility
        - max_drawdown: maximum peak-to-trough decline
    """
    prices_clean = prices.dropna()
    if len(prices_clean) < 2:
        return None

    # Returns
    daily_returns = prices_clean.pct_change().dropna()
    total_return = (prices_clean.iloc[-1] - prices_clean.iloc[0]) / prices_clean.iloc[0]

    # CAGR (Compound Annual Growth Rate)
    n_years = len(prices_clean) / 252  # 252 trading days per year
    cagr = (prices_clean.iloc[-1] / prices_clean.iloc[0]) ** (1 / n_years) - 1

    # Volatility
    daily_vol = daily_returns.std()
    annual_vol = daily_vol * np.sqrt(252)

    # Sharpe Ratio
    excess_return = cagr - risk_free_rate
    sharpe = excess_return / annual_vol if annual_vol > 0 else 0

    # Maximum Drawdown
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    # Find when max drawdown occurred
    max_dd_idx = drawdown.idxmin()  # trough date
    running_max_at_trough = running_max.loc[max_dd_idx]  # peak cumulative return value
    
    # Find the most recent date at or before trough where cumulative return reached that peak
    cum_before_trough = cumulative[cumulative.index <= max_dd_idx]
    peak_idx = cum_before_trough[cum_before_trough >= running_max_at_trough * 0.9999].index[-1] if len(cum_before_trough[cum_before_trough >= running_max_at_trough * 0.9999]) > 0 else cum_before_trough.idxmax()
    
    # Get the highest and lowest prices
    peak_price = prices_clean.loc[peak_idx]
    trough_price = prices_clean.loc[max_dd_idx]
    price_drop_pct = (trough_price - peak_price) / peak_price * 100
    
    max_dd_period = f"{peak_idx.strftime('%Y-%m-%d')} (${peak_price:.2f}) to {max_dd_idx.strftime('%Y-%m-%d')} (${trough_price:.2f}), {price_drop_pct:.2f}%" if hasattr(peak_idx, 'strftime') else f"{peak_idx} (${peak_price:.2f}) to {max_dd_idx} (${trough_price:.2f}), {price_drop_pct:.2f}%"

    # Count days with drawdown <= -20%
    severe_drawdown_days = (drawdown <= -0.20).sum()


    return {
        "total_return": total_return,
        "annual_return": cagr,
        "daily_volatility": daily_vol,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "max_drawdown_period": max_dd_period,
        "severe_drawdown_days": severe_drawdown_days
    }


def calendar_year_returns(prices: pd.DataFrame, years: list = None) -> pd.DataFrame:
    """% return per ticker for each requested calendar year.

    The base price for a year is the last available close on or before Dec 31
    of the prior year (falling back to that year's own first available close
    if there's no prior-year data, e.g. the ticker listed that year). For the
    current, still-in-progress year this yields a year-to-date return.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical prices indexed by date, one column per ticker.
    years : list of int, optional
        Calendar years to include. Defaults to every calendar year spanned by
        `prices` (its earliest year through its latest year).

    Returns
    -------
    pd.DataFrame indexed by year, one column per ticker.
    """
    if years is None:
        years = list(range(prices.index.min().year, prices.index.max().year + 1))

    rows = {}
    for year in years:
        row = {}
        for ticker in prices.columns:
            s = prices[ticker].dropna()
            s_year = s[s.index.year == year]
            if s_year.empty:
                continue
            prior = s[s.index < s_year.index[0]]
            base = prior.iloc[-1] if not prior.empty else s_year.iloc[0]
            row[ticker] = (s_year.iloc[-1] / base - 1) * 100
        rows[year] = row
    return pd.DataFrame(rows).T
