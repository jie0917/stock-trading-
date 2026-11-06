import numpy as np
import pandas as pd

from regression_testing.regression import beta_regression, trend_regression


def test_trend_regression_recovers_slope():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    prices = pd.Series(100 + 0.5 * np.arange(100), index=dates)

    model, fitted = trend_regression(prices)

    assert abs(model.params[1] - 0.5) < 1e-6
    assert abs(fitted.iloc[-1] - prices.iloc[-1]) < 1e-6


def test_beta_regression_recovers_beta_and_alpha():
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    x = pd.Series(np.random.RandomState(0).normal(0, 1, 200), index=dates)
    y = pd.Series(0.002 + 1.3 * x.values, index=dates)

    model, fitted = beta_regression(y, x)

    assert abs(model.params["x"] - 1.3) < 1e-6
    assert abs(model.params["const"] - 0.002) < 1e-6
    assert fitted.index.equals(dates)
