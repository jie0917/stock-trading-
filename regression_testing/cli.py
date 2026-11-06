"""Command-line entry point.

Example:
    python -m regression_testing.cli QQQ SPY --benchmark SPY --years 5
"""
import argparse

from .data import fetch_prices
from .plotting import plot_cumulative_returns, plot_price_trend, plot_return_scatter
from .regression import returns_beta, trend_regression


def main():
    parser = argparse.ArgumentParser(description="Run regression tests on US stock prices.")
    parser.add_argument("tickers", nargs="+", help="Tickers to analyze, e.g. QQQ SPY")
    parser.add_argument("--benchmark", default=None, help="Benchmark ticker for beta regression, e.g. SPY")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--years", type=int, default=5, help="Lookback years if --start not given")
    parser.add_argument("--no-show", action="store_true", help="Skip displaying plots (still computes regressions)")
    args = parser.parse_args()

    all_tickers = list(dict.fromkeys(args.tickers + ([args.benchmark] if args.benchmark else [])))
    prices = fetch_prices(all_tickers, start=args.start, end=args.end, years=args.years)
    show = not args.no_show

    for ticker in args.tickers:
        model, fitted = trend_regression(prices[ticker])
        print(f"\n=== {ticker} price trend regression ===")
        print(f"slope: {model.params[1]:.4f}  R^2: {model.rsquared:.4f}")
        plot_price_trend(prices[ticker], fitted, ticker, show=show)

        if args.benchmark and ticker != args.benchmark:
            beta_model, beta_fitted = returns_beta(prices, ticker, args.benchmark)
            print(f"\n=== {ticker} vs {args.benchmark} beta regression ===")
            print(f"beta: {beta_model.params['x']:.4f}  alpha: {beta_model.params['const']:.6f}  R^2: {beta_model.rsquared:.4f}")
            rets = prices[[ticker, args.benchmark]].pct_change().dropna()
            plot_return_scatter(rets[ticker], rets[args.benchmark], beta_fitted, ticker, args.benchmark, show=show)

    plot_cumulative_returns(prices, show=show)


if __name__ == "__main__":
    main()
