# stock-trading

This repo mainly focuses on stock trading algorithm analysis.

## ETF strategy (`etf_rotation_strategy.py`)

20-year backtest of a QQQ / QLD / BOXX strategy aiming for high return with low drawdown
(main strategy: `Ladder+QLD+TrendGuard+VolTgt25`):

- Trim QQQ into BOXX on new 52-week highs (+5% steps), buy back on -10/-15/-20% pullbacks.
- Add QLD tranches at -20/-30% drawdowns, take profit at +20% (proceeds go to QQQ).
- Trend guard (7-signal trend score, threshold 3/7), 25% vol target, QQQ never below 30%.

```bash
python etf_rotation_strategy.py   # backtest table + plot
python current_trend_score.py     # today's QQQ trend score
```

## Regression testing

Regression-tests US stock prices (e.g. QQQ, SPY) and plots the results using pandas.

Two kinds of regression:
- **Trend regression** — OLS fit of price against time, to see if a stock is tracking its long-run trend.
- **Beta regression** — OLS fit of one ticker's daily returns against a benchmark's (e.g. QQQ vs. SPY), giving alpha, beta, and R².

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the example script (QQQ vs. SPY, 5-year lookback):

```bash
python main.py
```

Or use the CLI for arbitrary tickers:

```bash
python -m regression_testing.cli QQQ SPY AAPL --benchmark SPY --years 5
```

Options: `--start YYYY-MM-DD`, `--end YYYY-MM-DD`, `--years N`, `--no-show` (skip displaying plots).

## Tests

```bash
pytest
```
