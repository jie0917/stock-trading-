"""Example: plot cumulative returns for all configured tickers with performance metrics."""
from regression_testing.data import fetch_prices, align_to_common_start
from regression_testing.dca import simulate_dca
from regression_testing.plotting import plot_cumulative_returns, plot_dca_growth, plot_calendar_year_returns
from regression_testing.regression import calculate_performance_metrics, calendar_year_returns

TICKERS = ["QQQ", "SPY", "VYM", "JEPQ", "JEPI", "SPYI", "QQQI", "SCHD", "BRK-B", "DIVO"]
ALIGN_DCA = False  # True: truncate all tickers to the latest common start date for metrics + backtesting.

if __name__ == "__main__":
    prices = fetch_prices(TICKERS, years=20)
    metrics_prices = align_to_common_start(prices) if ALIGN_DCA else prices

    for ticker in TICKERS:
        metrics = calculate_performance_metrics(metrics_prices[ticker])
        print(f"\n=== {ticker} Performance Metrics ===")
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
    

    
    plot_cumulative_returns(prices)

    plot_calendar_year_returns(calendar_year_returns(metrics_prices))

    # DCA backtest: $1,000/month into each ticker independently, each starting
    # from its own available history (no common-start alignment).
    dca = simulate_dca(prices, monthly_investment=1000, align=ALIGN_DCA)
    history_note = "aligned to common start date" if ALIGN_DCA else "each from its own available history"
    print(f"\n=== DCA: $1,000/month per ticker ({history_note}) ===")
    for ticker, s in dca["summary"].items():
        if ticker == "Total":
            continue
        print(f"{ticker}: invested ${s['total_invested']:,.0f} -> ${s['final_value']:,.0f} "
              f"({s['total_return_pct']:.1f}%, {s['num_purchases']} purchases)")
    total = dca["summary"]["Total"]
    print(f"\nCombined (all {len(TICKERS)} tickers): invested ${total['total_invested']:,.0f} -> "
          f"${total['final_value']:,.0f} ({total['total_return_pct']:.1f}%)")

    plot_dca_growth(dca["value"], dca["invested"])
