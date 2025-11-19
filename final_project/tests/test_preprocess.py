"""Test preprocessing functions."""

import numpy as np
import pandas as pd
import pytest

from src.vn50.data import preprocess, validation


def test_validate_raw_prices_pass():
    """Test validation passes for valid data."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    prices = pd.DataFrame(
        np.random.rand(100, 3) * 100 + 10,
        index=dates,
        columns=["A", "B", "C"],
    )

    # Should not raise
    validation.validate_raw_prices(
        prices, expected_tickers=["A", "B", "C"], start_date="2020-01-01", end_date="2020-04-09"
    )


def test_validate_raw_prices_duplicate_dates():
    """Test validation fails for duplicate dates."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    dates = list(dates) + [dates[0]]  # Add duplicate
    prices = pd.DataFrame(
        np.random.rand(101, 3) * 100 + 10,
        index=dates,
        columns=["A", "B", "C"],
    )

    with pytest.raises(ValueError, match="duplicate dates"):
        validation.validate_raw_prices(prices)


def test_validate_raw_prices_missing_tickers():
    """Test validation fails for missing tickers."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    prices = pd.DataFrame(
        np.random.rand(100, 2) * 100 + 10,
        index=dates,
        columns=["A", "B"],
    )

    with pytest.raises(ValueError, match="Missing expected tickers"):
        validation.validate_raw_prices(prices, expected_tickers=["A", "B", "C"])


def test_winsorize_returns_no_lookahead():
    """Test winsorization does not look ahead."""
    # Create synthetic data with known outlier at t=100
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    returns = pd.DataFrame(
        np.random.randn(200, 1) * 0.01,
        index=dates,
        columns=["A"],
    )
    # Add outlier at t=100
    returns.loc[returns.index[100], "A"] = 0.5  # Large positive return

    # Winsorize
    winsorized = preprocess.winsorize_returns(
        returns, mode="rolling_quantile", window=60, lower_q=0.01, upper_q=0.99
    )

    # Check that quantile at t=100 is calculated from t=0 to t=99 (shifted)
    # The outlier at t=100 should not affect the quantile used to clip it
    q_at_100 = (
        returns["A"]
        .shift(1)
        .rolling(window=60, min_periods=25)
        .quantile(0.99)
        .iloc[100]
    )

    # The clipped value at t=100 should be <= q_at_100
    assert winsorized.loc[winsorized.index[100], "A"] <= q_at_100 or np.isnan(
        winsorized.loc[winsorized.index[100], "A"]
    )


def test_mask_stale_returns():
    """Test stale returns masking."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    returns = pd.DataFrame(
        {
            "A": [0.01, 0, 0, 0, 0.02, 0.01, 0, 0, 0, 0.01],  # 4 zeros at start, 4 at end
            "B": [0.01, 0.02, 0.01, 0, 0, 0.01, 0.02, 0.01, 0.02, 0.01],  # Only 2 zeros
        },
        index=dates,
    )

    masked = preprocess.mask_stale_returns(returns, min_consecutive_days=3)

    # A should have zeros masked from position 3 onwards (first run) and from position 7 onwards (second run)
    # B should not be masked (only 2 consecutive zeros)

    # Check that B is not masked
    assert not masked["B"].isna().any()

    # Check that A has some NaNs (masked stale returns)
    assert masked["A"].isna().any()


def test_load_trading_calendar_not_found():
    """Test calendar loading when file not found."""
    result = preprocess.load_trading_calendar("nonexistent_file.csv")
    assert result is None

