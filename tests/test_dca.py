import numpy as np
import pandas as pd

from regression_testing.dca import simulate_dca


def test_simulate_dca_flat_price_matches_shares_times_price():
    # Constant $100 price for 2 years of daily data -> each month buys 10 shares
    # ($1,000 / $100), so value should always equal (num purchases so far) * 1000.
    dates = pd.date_range("2022-01-01", "2023-12-31", freq="D")
    prices = pd.DataFrame({"FLAT": 100.0}, index=dates)

    result = simulate_dca(prices, monthly_investment=1000)
    value = result["value"]
    invested = result["invested"]
    summary = result["summary"]["FLAT"]

    # 24 calendar months of contributions.
    assert summary["num_purchases"] == 24
    assert summary["total_invested"] == 24000
    # Flat price -> no gain, final value equals total invested.
    assert abs(summary["final_value"] - 24000) < 1e-6
    assert abs(summary["gain"]) < 1e-6
    assert abs(summary["total_return_pct"]) < 1e-6

    # Value should never exceed cumulative invested when price is flat.
    assert (value["FLAT"] <= invested["FLAT"] + 1e-6).all()
    # Nothing invested before the first purchase date.
    assert value["FLAT"].iloc[0] == 1000  # first day of the series is itself a purchase


def test_simulate_dca_total_column_sums_tickers():
    dates = pd.date_range("2022-01-01", "2022-06-30", freq="D")
    prices = pd.DataFrame(
        {
            "A": 100.0 + 0.1 * np.arange(len(dates)),
            "B": 50.0 + 0.05 * np.arange(len(dates)),
        },
        index=dates,
    )

    result = simulate_dca(prices, monthly_investment=1000)
    value, invested, summary = result["value"], result["invested"], result["summary"]

    assert np.allclose(value["Total"], value["A"] + value["B"])
    assert np.allclose(invested["Total"], invested["A"] + invested["B"])
    assert summary["Total"]["total_invested"] == summary["A"]["total_invested"] + summary["B"]["total_invested"]
