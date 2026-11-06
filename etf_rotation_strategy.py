"""ETF trading strategies over QQQ / SPY / QLD, with BOXX as cash.

Goal: high return with low drawdown, backtested over the last 20 years.

Main strategy — "Ladder + QLD + Trend Guard + Vol Target" (ladder_trend_weights
with MAIN_LADDER settings). "High" means QQQ's 52-week closing high.
  1. Scale out on strength: each time QQQ makes a new high 5% above the last
     trim level, move 10% of the portfolio from QQQ into DEFENSIVE_BASKET
     (BOXX by default), keeping at least 40% in QQQ.
  2. Scale back in on pullbacks: the defensive money is moved back into QQQ in
     thirds as QQQ's drawdown from its high reaches -10%, -15% and -20%.
  3. Add leverage on deep dips: at -20% and -30% drawdown, 15% of the
     portfolio each is bought in QLD (2x QQQ), paid for from BOXX first and by
     selling QQQ only for the rest. Each tranche is sold, back to where its
     money came from, once QLD is up 20% from that tranche's entry price.
  4. Trend guard: dips are only bought (and QLD only held) while QQQ's uptrend
     is intact. When 2 or fewer of 7 trend signals (price above its
     50/100/150/200-day SMA, above its price 3/6/12 months ago) are positive,
     QLD goes back to QQQ, QQQ is scaled by the signal score and the rest goes
     to BROKEN_TREND_BASKET. QQQ never goes below 30% of the portfolio.
  5. Volatility target (25%): QQQ-equivalent exposure (QQQ + 2 x QLD) is scaled
     down when QQQ's 20-day realized vol is high, which cuts fast crashes
     (e.g. Feb-Mar 2020).

Alternative — "HighTrim + Trend Guard + Vol Target" (trim_trend_weights):
same trim on new highs, but buys everything back at once at -10% and never
uses QLD. A bit less return for a bit less drawdown.

For comparison it also runs the unguarded versions, a pure trend-ensemble
strategy (trend_vol_weights), a monthly dual-momentum rotation and
buy-and-hold benchmarks.

Signals are computed at the close of day t and executed at the close of day
t+1 (one-day lag, no lookahead). Transaction cost is charged on turnover.

Several ETFs are younger than the backtest window, so their returns are
spliced from older stand-ins before inception (see PROXIES):
  - BOXX (Dec 2022) <- BIL 1-3 month T-bill ETF (May 2007) <- 13-week T-bill
    yield (^IRX), since BOXX is designed to track T-bill returns.
"""
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regression_testing.data import fetch_prices

ROTATION_ETFS = ["SPY", "QQQ"]
LEVERAGED_ETF = "QLD"  # 2x QQQ, only bought on deep QQQ drawdowns
RISK_ETFS = ROTATION_ETFS + [LEVERAGED_ETF]
CASH_ETF = "BOXX"
PORTFOLIO_ETFS = RISK_ETFS + [CASH_ETF]
TBILL_YIELD = "^IRX"  # 13-week T-bill yield in %, turned into a daily return
# Older stand-ins whose returns fill in before each ETF's inception, in order.
PROXIES = {CASH_ETF: ["BIL", TBILL_YIELD]}

BACKTEST_YEARS = 20
WARMUP_DAYS = 400  # extra history for the 12-month momentum / 200-day SMA
COST_BPS = 5  # per unit of turnover
RISK_FREE = 0.02

# Trend Ensemble + Vol Target parameters
RISK_ALLOC = {"QQQ": 1.0}  # base risk sleeve, e.g. {"QQQ": 0.7, "SPY": 0.3}
SMA_WINDOWS = (50, 100, 150, 200)
MOMENTUM_MONTHS = (3, 6, 12)
VOL_TARGET = 0.20  # annualized
VOL_LOOKBACK = 20
REBALANCE_THRESHOLD = 0.10  # sum of absolute weight changes

# High-trim (take profit on new highs, buy back on pullbacks) parameters
# Where trimmed QQQ goes while the trend is intact, and where the risk money
# goes once the trend guard trips (*_trend_weights only).
# BOXX alone beat SPY/BOXX mixes on drawdown and Sharpe at about the same CAGR.
DEFENSIVE_BASKET = {CASH_ETF: 1.0}
BROKEN_TREND_BASKET = DEFENSIVE_BASKET

# Main strategy: ladder_trend_weights(prices, vol_target=0.25, **MAIN_LADDER)
# min_score=3/7: the trend guard only cuts QQQ once the score is 2/7 or lower.
# min_qqq_hold=0.30: QQQ never goes below 30% of the portfolio.
MAIN_LADDER = dict(min_score=3 / 7, min_qqq_hold=0.30, peak_window=252, up_step=0.05,
                   buyback_levels=(0.10, 0.15, 0.20), qld_levels=(0.20, 0.30),
                   qld_sale_to_qqq=True)

# Rotation (comparison) parameters
TOP_N = 1
SMA_DAYS = 200
BAND = 0.02  # daily guard triggers when SPY < SMA200 * (1 - BAND)


def load_prices() -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=365 * BACKTEST_YEARS + WARMUP_DAYS)
    proxies = [p for chain in PROXIES.values() for p in chain]
    prices = fetch_prices(PORTFOLIO_ETFS + proxies, start=start, end=end)

    rets = prices.pct_change(fill_method=None)
    rets[TBILL_YIELD] = prices[TBILL_YIELD].ffill().shift(1) / 100 / 252
    for ticker, chain in PROXIES.items():
        spliced = rets[ticker]
        for proxy in chain:
            spliced = spliced.fillna(rets[proxy])
        prices[ticker] = (1 + spliced.fillna(0)).cumprod() * 100
    return prices.drop(columns=proxies).ffill()


def trend_score(price: pd.Series) -> pd.Series:
    """Fraction (0..1) of SMA and momentum trend signals that are positive."""
    signals = [price > price.rolling(n).mean() for n in SMA_WINDOWS]
    signals += [price > price.shift(21 * m) for m in MOMENTUM_MONTHS]
    return sum(s.astype(float) for s in signals) / len(signals)


def trend_vol_weights(prices: pd.DataFrame, alloc=None, vol_target=VOL_TARGET) -> pd.DataFrame:
    """Target weights for the Trend Ensemble + Vol Target strategy."""
    alloc = alloc or RISK_ALLOC
    cols = PORTFOLIO_ETFS
    rets = prices.pct_change()

    target = pd.DataFrame(0.0, index=prices.index, columns=cols)
    for t, base in alloc.items():
        target[t] = base * trend_score(prices[t])

    if vol_target:
        sleeve_rets = sum(rets[t] * base for t, base in alloc.items())
        vol = sleeve_rets.rolling(VOL_LOOKBACK).std() * np.sqrt(252)
        scale = (vol_target / vol).clip(upper=1).fillna(0)
        target[list(alloc)] = target[list(alloc)].mul(scale, axis=0)
    target[CASH_ETF] = 1 - target[RISK_ETFS].sum(axis=1)

    # Hold current weights until the target drifts far enough to justify a trade.
    values = target.values
    held = np.empty_like(values)
    current = values[0]
    for i, row in enumerate(values):
        if np.abs(row - current).sum() > REBALANCE_THRESHOLD:
            current = row
        held[i] = current
    return pd.DataFrame(held, index=target.index, columns=cols)


def momentum_score(prices: pd.DataFrame) -> pd.DataFrame:
    """13612W momentum: (12*r1 + 4*r3 + 2*r6 + 1*r12) / 4, months = 21 trading days."""
    r = {m: prices / prices.shift(21 * m) - 1 for m in (1, 3, 6, 12)}
    return (12 * r[1] + 4 * r[3] + 2 * r[6] + r[12]) / 4


def month_end_mask(index: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(index.month, index=index)
    return s != s.shift(-1)


def rotation_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """Monthly dual-momentum rotation with a daily SPY 200-day SMA guard."""
    score = momentum_score(prices)
    sma = prices.rolling(SMA_DAYS).mean()
    is_month_end = month_end_mask(prices.index)

    cols = PORTFOLIO_ETFS
    weights = pd.DataFrame(np.nan, index=prices.index, columns=cols)
    current = pd.Series(0.0, index=cols)
    current[CASH_ETF] = 1.0
    guard_on = False

    for i, date in enumerate(prices.index):
        if i < SMA_DAYS + 252:
            weights.iloc[i] = current
            continue

        spy_below = prices.at[date, "SPY"] < sma.at[date, "SPY"] * (1 - BAND)
        if spy_below and not guard_on:
            guard_on = True
            current = pd.Series(0.0, index=cols)
            current[CASH_ETF] = 1.0

        if is_month_end[date]:
            guard_on = spy_below
            current = pd.Series(0.0, index=cols)
            if guard_on:
                current[CASH_ETF] = 1.0
            else:
                s = score.loc[date, ROTATION_ETFS].dropna()
                eligible = s[(s > score.at[date, CASH_ETF])
                             & (prices.loc[date, s.index] > sma.loc[date, s.index])]
                for t in eligible.sort_values(ascending=False).index[:TOP_N]:
                    current[t] = 1.0 / TOP_N
                current[CASH_ETF] = 1.0 - current[RISK_ETFS].sum()

        weights.iloc[i] = current
    return weights


def high_trim_weights(prices: pd.DataFrame, up_step=0.03, trim=0.10, min_qqq=0.40,
                      buyback_dd=0.05, defensive=None) -> pd.DataFrame:
    """Take profit on QQQ strength, buy back on QQQ pullbacks.

    - Each time QQQ makes a new high at least `up_step` above the last trim
      level, move `trim` of the portfolio from QQQ into the defensive basket
      (QQQ never goes below `min_qqq`).
    - When QQQ falls `buyback_dd` below its peak, sell the defensive basket and
      go back to 100% QQQ. The next trim then needs a new high `up_step` above
      the old peak.
    """
    defensive = defensive or DEFENSIVE_BASKET
    cols = PORTFOLIO_ETFS
    qqq = prices["QQQ"].values

    weights = np.zeros((len(qqq), len(cols)))
    qqq_col = cols.index("QQQ")
    w_qqq, peak, trim_level = 1.0, qqq[0], qqq[0]
    for i, px in enumerate(qqq):
        if px > peak:
            peak = px
            if px >= trim_level * (1 + up_step) and w_qqq > min_qqq:
                w_qqq = max(min_qqq, w_qqq - trim)
                trim_level = px
        elif px <= peak * (1 - buyback_dd) and w_qqq < 1.0:
            w_qqq = 1.0
            trim_level = peak

        weights[i, qqq_col] = w_qqq
        for t, share in defensive.items():
            weights[i, cols.index(t)] += (1 - w_qqq) * share
    return pd.DataFrame(weights, index=prices.index, columns=cols)


def trim_trend_weights(prices: pd.DataFrame, min_score=4 / 7, vol_target=VOL_TARGET,
                       broken_basket=None, **trim_kwargs) -> pd.DataFrame:
    """high_trim_weights plus a trend guard and vol target.

    Buying the dip only works while QQQ's uptrend is intact; in a real bear
    market (2022) it means riding the whole decline. So when QQQ's trend_score
    drops below `min_score`, the QQQ weight is scaled by the score and the
    defensive money goes to `broken_basket` (default BROKEN_TREND_BASKET).
    The QQQ weight is also capped by the vol target, then rounded to 10%
    steps to avoid daily micro-trades.
    """
    defensive = trim_kwargs.pop("defensive", None) or DEFENSIVE_BASKET
    base = high_trim_weights(prices, defensive=defensive, **trim_kwargs)
    score = trend_score(prices["QQQ"])
    broken = score < min_score

    qqq = base["QQQ"].where(~broken, base["QQQ"] * score)
    if vol_target:
        vol = prices["QQQ"].pct_change().rolling(VOL_LOOKBACK).std() * np.sqrt(252)
        qqq = qqq * (vol_target / vol).clip(upper=1).fillna(0)
    qqq = qqq.round(1)

    weights = pd.DataFrame(0.0, index=prices.index, columns=base.columns)
    weights["QQQ"] = qqq
    for t, share in defensive.items():
        weights[t] += (1 - qqq).where(~broken, 0) * share
    for t, share in (broken_basket or BROKEN_TREND_BASKET).items():
        weights[t] += (1 - qqq).where(broken, 0) * share
    return weights


def ladder_weights(prices: pd.DataFrame, up_step=0.03, trim=0.10, min_qqq=0.40,
                   buyback_levels=(0.05, 0.10, 0.15), buyback_fraction=1.0,
                   qld_levels=(0.15, 0.20, 0.25), qld_step=0.15, qld_take_profit=0.20,
                   qld_funding="defensive_then_qqq", qld_sale_to_qqq=False,
                   peak_window=None, defensive=None) -> pd.DataFrame:
    """Scale out of QQQ on new highs, scale back in on pullbacks, add QLD on deep dips.

    - "High" and drawdowns are measured against the highest close over the last
      `peak_window` trading days (252 = 52-week high), or since the start of
      the data when `peak_window` is None. With a window, the trim reference
      (last trim price, or the old high after a buyback) is reset to the
      window's high once it is older than the window.

    - Trim (same as high_trim_weights): each new high `up_step` above the last
      trim level moves `trim` of the portfolio from QQQ to the defensive basket.
    - Buy back in tranches: the defensive weight held when the pullback started
      is moved back into QQQ in equal parts as the drawdown from the peak
      reaches each of `buyback_levels` (e.g. 1/3 at -5%, 1/3 at -10%, 1/3 at -15%).
      `buyback_fraction` < 1 moves only that share back, keeping the rest in
      the defensive basket (e.g. as dry powder for QLD).
    - QLD in tranches: at each of `qld_levels` drawdown, buy `qld_step` of the
      portfolio in QLD. Each tranche is sold once QLD is up `qld_take_profit`
      from that tranche's entry price, so QLD is taken off gradually on the way
      up. A sold tranche is bought again only after QQQ has climbed back above
      its level and then falls to it again.
    - `qld_funding` says where QLD money comes from (and returns to on sale):
      "qqq" sells QQQ; "defensive" uses only the defensive basket, so QLD is
      capped by what is left there; "defensive_then_qqq" uses the defensive
      basket first and sells QQQ for the rest.
    - `qld_sale_to_qqq`: when a QLD tranche is sold, put the part that came from
      the defensive basket into QQQ instead of back into the defensive basket.
    """
    defensive = defensive or DEFENSIVE_BASKET
    cols = PORTFOLIO_ETFS
    qqq = prices["QQQ"].values
    prior = prices["QQQ"].shift(1)
    prior_high = (prior.rolling(peak_window, min_periods=1).max() if peak_window
                  else prior.cummax()).values  # highest close before today

    weights = np.zeros((len(qqq), len(cols)))
    w_def, trim_level, trim_set_at = 0.0, qqq[0], 0
    cycle_def, bought_back = 0.0, 0
    qld_px = prices[LEVERAGED_ETF].values
    qld_entry = [None] * len(qld_levels)  # QLD entry price per tranche, None = not held
    qld_armed = [True] * len(qld_levels)  # False after a sale until dd is back above the level
    from_def = 0.0
    for i, px in enumerate(qqq):
        peak = px if np.isnan(prior_high[i]) else max(prior_high[i], px)
        if peak_window and i - trim_set_at > peak_window:
            # The trim reference is older than the window: reset it to the
            # window's high so an old high no longer blocks trimming.
            trim_level, trim_set_at = prior_high[i], i
        if px > prior_high[i]:  # new high (False on day 0, where prior_high is NaN)
            cycle_def, bought_back = w_def, 0
            if px >= trim_level * (1 + up_step) and 1 - w_def > min_qqq:
                w_def = min(1 - min_qqq, w_def + trim)
                cycle_def = w_def
                trim_level, trim_set_at = px, i

        dd = 1 - px / peak
        while bought_back < len(buyback_levels) and dd >= buyback_levels[bought_back]:
            bought_back += 1
            w_def = min(w_def, cycle_def * (1 - buyback_fraction * bought_back / len(buyback_levels)))
            if peak > trim_level:  # re-trim only above the old peak
                trim_level, trim_set_at = peak, i

        for k, level in enumerate(qld_levels):
            if qld_entry[k] is None:
                if dd < level:
                    qld_armed[k] = True
                elif qld_armed[k]:
                    qld_entry[k] = qld_px[i]
            elif np.isnan(qld_entry[k]):  # signal fired before QLD's launch
                qld_entry[k] = qld_px[i]
            elif qld_px[i] >= qld_entry[k] * (1 + qld_take_profit):
                qld_entry[k], qld_armed[k] = None, False
        w_qld = qld_step * sum(e is not None for e in qld_entry)
        if qld_sale_to_qqq and from_def > w_qld:
            # Defensive money freed by today's QLD sale moves into QQQ.
            released = from_def - w_qld
            w_def -= released
            cycle_def = max(0.0, cycle_def - released)
        if qld_funding == "qqq":
            from_def = 0.0
        elif qld_funding == "defensive":
            w_qld = from_def = min(w_qld, w_def)
        elif qld_funding == "defensive_then_qqq":
            from_def = min(w_qld, w_def)
        else:
            raise ValueError(f"unknown qld_funding: {qld_funding}")
        w_qld = min(w_qld, 1 - w_def + from_def)

        weights[i, cols.index("QQQ")] = 1 - w_def + from_def - w_qld
        weights[i, cols.index(LEVERAGED_ETF)] = w_qld
        for t, share in defensive.items():
            weights[i, cols.index(t)] += (w_def - from_def) * share
    return pd.DataFrame(weights, index=prices.index, columns=cols)


def ladder_trend_weights(prices: pd.DataFrame, min_score=4 / 7, vol_target=VOL_TARGET,
                         min_qqq_hold=0.0, score_floors=None, broken_basket=None,
                         **ladder_kwargs) -> pd.DataFrame:
    """ladder_weights plus the same trend guard and vol target as trim_trend_weights.

    When QQQ's trend is broken, QLD is swapped back to QQQ (no leverage in a
    downtrend), QQQ is scaled by the trend score and the rest goes to BOXX.
    The vol target caps the QQQ-equivalent exposure (QQQ + 2 x QLD).
    QQQ never goes below `min_qqq_hold`, nor below `score_floors[k]` once at
    least k of the 7 trend signals are positive (e.g. {4: 0.5, 5: 0.7}); the
    top-up comes out of the defensive money first and out of QLD only if that
    is not enough.
    """
    defensive = ladder_kwargs.pop("defensive", None) or DEFENSIVE_BASKET
    base = ladder_weights(prices, defensive=defensive, **ladder_kwargs)
    score = trend_score(prices["QQQ"])
    broken = score < min_score

    qld = base[LEVERAGED_ETF].where(~broken, 0.0)
    qqq = base["QQQ"].where(~broken, (base["QQQ"] + base[LEVERAGED_ETF]) * score)
    if vol_target:
        vol = prices["QQQ"].pct_change().rolling(VOL_LOOKBACK).std() * np.sqrt(252)
        exposure = (qqq + 2 * qld).replace(0, np.nan)
        scale = (vol_target / (vol * exposure)).clip(upper=1).fillna(0)
        qqq, qld = qqq * scale, qld * scale
    qqq, qld = qqq.round(1), qld.round(2)
    floor = pd.Series(min_qqq_hold, index=prices.index)
    positives = (score * (len(SMA_WINDOWS) + len(MOMENTUM_MONTHS))).round()
    for k, level in (score_floors or {}).items():
        floor = floor.where(positives < k, np.maximum(floor, level))
    top_up = (floor - qqq).clip(lower=0)
    qld = qld - (top_up - top_up.clip(upper=1 - qqq - qld))
    qqq = qqq + top_up

    weights = pd.DataFrame(0.0, index=prices.index, columns=base.columns)
    weights["QQQ"] = qqq
    weights[LEVERAGED_ETF] = qld
    rest = 1 - qqq - qld
    for t, share in defensive.items():
        weights[t] += rest.where(~broken, 0) * share
    for t, share in (broken_basket or BROKEN_TREND_BASKET).items():
        weights[t] += rest.where(broken, 0) * share
    return weights


def deep_guard_weights(prices: pd.DataFrame, exit_score=0.0, reenter_score=4 / 7,
                       broken_basket=None, **ladder_kwargs) -> pd.DataFrame:
    """ladder_weights with a low-turnover crash guard.

    Follows ladder_weights (trim on new highs, buy dips) and only steps aside
    in a deep breakdown: when QQQ's trend_score falls to `exit_score` or below,
    everything moves to `broken_basket` (default BOXX) and stays there until
    the score recovers to `reenter_score`. The gap between the two thresholds
    avoids flipping in and out around a single level.
    """
    base = ladder_weights(prices, **ladder_kwargs)
    score = trend_score(prices["QQQ"]).values

    out = np.zeros(len(score), dtype=bool)
    is_out = False
    for i, s in enumerate(score):
        if not is_out and s <= exit_score + 1e-9:
            is_out = True
        elif is_out and s >= reenter_score - 1e-9:
            is_out = False
        out[i] = is_out

    weights = base.copy()
    weights.loc[out, :] = 0.0
    for t, share in (broken_basket or BROKEN_TREND_BASKET).items():
        weights.loc[out, t] = share
    return weights


def static_weights(prices: pd.DataFrame, alloc: dict) -> pd.DataFrame:
    """Constant target weights, bought once and held (no rebalancing)."""
    w = pd.DataFrame(0.0, index=prices.index, columns=list(alloc))
    for t, v in alloc.items():
        w[t] = v
    return w


def run_backtest(prices: pd.DataFrame, weights: pd.DataFrame, start) -> pd.Series:
    """Equity curve (starting at 1.0 on `start`) for a table of target weights.

    The portfolio rebalances to the target only on days the target changes and
    otherwise lets positions drift with the market, so constant weights mean
    true buy-and-hold. Targets set at day t's close are traded at t+1's close.
    """
    target = weights.shift(1).loc[start:].fillna(0)
    rets = prices[weights.columns].pct_change().loc[start:].fillna(0).values
    changed = (target.diff().abs().sum(axis=1) > 1e-9).values
    target = target.values

    positions = target[0].copy()  # dollar value per ETF, total starts at 1.0
    equity = np.empty(len(target))
    for i in range(len(target)):
        if i > 0:
            positions = positions * (1 + rets[i])
        total = positions.sum()
        if changed[i]:
            turnover = np.abs(target[i] - positions / total).sum()
            total *= 1 - turnover * COST_BPS / 10_000
            positions = target[i] * total
        equity[i] = total
    return pd.Series(equity, index=weights.loc[start:].index)


def metrics(equity: pd.Series) -> dict:
    daily = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = daily.std() * np.sqrt(252)
    dd = equity / equity.cummax() - 1
    max_dd = dd.min()
    downside = daily[daily < 0].std() * np.sqrt(252)
    return {
        "Total Return": equity.iloc[-1] / equity.iloc[0] - 1,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": (cagr - RISK_FREE) / vol if vol else np.nan,
        "Sortino": (cagr - RISK_FREE) / downside if downside else np.nan,
        "Max Drawdown": max_dd,
        "Calmar": cagr / abs(max_dd) if max_dd else np.nan,
        "Worst Year": equity.resample("YE").last().pct_change().min(),
    }


def print_table(results: dict):
    df = pd.DataFrame({name: metrics(eq) for name, eq in results.items()}).T
    pct_cols = ["Total Return", "CAGR", "Volatility", "Max Drawdown", "Worst Year"]
    fmt = df.copy().astype(object)
    for c in df.columns:
        fmt[c] = df[c].map(lambda v: f"{v:.1%}" if c in pct_cols else f"{v:.2f}")
    print(fmt.to_string())


def plot_results(results: dict, strategy_name: str, weights: pd.DataFrame, start):
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    for name, eq in results.items():
        lw = 2.2 if name == strategy_name else 1.0
        axes[0].plot(eq.index, eq * 10_000, label=name, linewidth=lw)
        axes[1].plot(eq.index, (eq / eq.cummax() - 1) * 100, label=name, linewidth=lw)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Value of $10,000 (log)")
    axes[0].set_title(f"ETF Strategy Backtest — last {BACKTEST_YEARS} years")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].set_ylabel("Drawdown %")

    w = weights.loc[start:]
    w = w.loc[:, (w > 0).any()]
    axes[2].stackplot(w.index, w.T.values, labels=w.columns)
    axes[2].set_ylabel(f"{strategy_name}\nweights")
    axes[2].set_ylim(0, 1)
    axes[2].legend(loc="upper left", fontsize=8, ncol=len(w.columns))
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_allocation(name: str, weights: pd.DataFrame, start):
    held = weights.loc[start:].shift(1)
    trades = (held.diff().abs().sum(axis=1) > 1e-9).sum()
    exposure = (held[RISK_ETFS].sum(axis=1) + held[LEVERAGED_ETF]).mean()  # QLD counts 2x
    print(f"\n{name}: {trades} rebalances, avg QQQ-equivalent exposure {exposure:.0%}")
    latest = weights.iloc[-1]
    print(f"Current allocation ({weights.index[-1].date()}):")
    for t, v in latest[latest > 0.001].sort_values(ascending=False).items():
        print(f"  {t}: {v:.1%}")


if __name__ == "__main__":
    prices = load_prices()
    start = prices.index[-1] - pd.DateOffset(years=BACKTEST_YEARS)
    print(f"Backtest: {prices.loc[start:].index[0].date()} -> {prices.index[-1].date()}")

    strategy_name = "Ladder+QLD+TrendGuard+VolTgt25"
    strategy_weights = ladder_trend_weights(prices, vol_target=0.25, **MAIN_LADDER)
    alt_name = "HighTrim+TrendGuard+VolTgt"
    alt_weights = trim_trend_weights(prices, buyback_dd=0.10)

    results = {
        strategy_name: run_backtest(prices, strategy_weights, start),
        alt_name: run_backtest(prices, alt_weights, start),
        "HighTrim (buy back @-5%)": run_backtest(prices, high_trim_weights(prices), start),
        "HighTrim (buy back @-10%)": run_backtest(
            prices, high_trim_weights(prices, buyback_dd=0.10), start),
        "Ladder+QLD+Guard (old: ATH, +3%, -5/-10/-15)": run_backtest(
            prices, ladder_trend_weights(prices, vol_target=0.25), start),
        "Ladder (no QLD)": run_backtest(prices, ladder_weights(prices, qld_levels=()), start),
        "ATH Trim + Buy -10/-20/-30": run_backtest(prices, ladder_weights(
            prices, buyback_levels=(0.10, 0.20, 0.30), qld_levels=()), start),
        "ATH Trim + Buy + DeepGuard": run_backtest(prices, deep_guard_weights(
            prices, buyback_levels=(0.10, 0.20, 0.30), qld_levels=()), start),
        "Ladder+QLD": run_backtest(prices, ladder_weights(prices), start),
        "Trend+VolTarget (QQQ/BOXX)": run_backtest(prices, trend_vol_weights(prices), start),
        "Momentum Rotation": run_backtest(prices, rotation_weights(prices), start),
        "SPY B&H": run_backtest(prices, static_weights(prices, {"SPY": 1.0}), start),
        "QQQ B&H": run_backtest(prices, static_weights(prices, {"QQQ": 1.0}), start),
        "QLD B&H": run_backtest(prices, static_weights(prices, {LEVERAGED_ETF: 1.0}), start),
        "60 QQQ / 40 BOXX": run_backtest(
            prices, static_weights(prices, {"QQQ": 0.6, CASH_ETF: 0.4}), start),
    }

    print()
    print_table(results)

    print_allocation(strategy_name, strategy_weights, start)
    print_allocation(alt_name, alt_weights, start)

    plot_names = [strategy_name, alt_name, "QQQ B&H", "SPY B&H", "60 QQQ / 40 BOXX"]
    plot_results({n: results[n] for n in plot_names}, strategy_name, strategy_weights, start)
