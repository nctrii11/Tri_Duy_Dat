"""Test returns calculation."""

import pandas as pd
import pytest

from src.vn50.features import returns


def test_make_returns_log():
    """Test log returns calculation."""
    prices = pd.DataFrame(
        {
            "A": [100, 110, 105, 120],
            "B": [50, 55, 52, 60],
        },
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )

    result = returns.make_returns(prices, kind="log", window=1, freq="D")

    assert len(result) == 3  # First row is NaN
    assert "A" in result.columns
    assert "B" in result.columns
    assert not result.isna().any().any()


def test_make_returns_arith():
    """Test arithmetic returns calculation."""
    prices = pd.DataFrame(
        {
            "A": [100, 110, 105, 120],
            "B": [50, 55, 52, 60],
        },
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )

    result = returns.make_returns(prices, kind="arith", window=1, freq="D")

    assert len(result) == 3
    assert "A" in result.columns
    assert "B" in result.columns


def test_make_returns_monthly():
    """Test monthly resampling."""
    prices = pd.DataFrame(
        {
            "A": [100, 110, 105, 120, 125, 130],
            "B": [50, 55, 52, 60, 58, 65],
        },
        index=pd.date_range("2020-01-01", periods=6, freq="D"),
    )

    result = returns.make_returns(prices, kind="arith", window=1, freq="M")

    assert len(result) <= len(prices)  # Monthly should have fewer periods

