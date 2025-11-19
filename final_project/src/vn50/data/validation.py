"""Validate raw price data."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_raw_prices(
    prices: pd.DataFrame,
    expected_tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """
    Validate raw price data.

    Args:
        prices: DataFrame with prices
        expected_tickers: List of expected ticker symbols
        start_date: Expected start date (YYYY-MM-DD)
        end_date: Expected end date (YYYY-MM-DD)

    Raises:
        ValueError: If validation fails
    """
    # Check index is DatetimeIndex
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError(f"Index must be DatetimeIndex, got {type(prices.index)}")

    # Check for duplicate dates
    if prices.index.duplicated().any():
        n_duplicates = prices.index.duplicated().sum()
        raise ValueError(f"Found {n_duplicates} duplicate dates in index")

    # Check index is sorted
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted in ascending order")

    # Check expected tickers
    if expected_tickers is not None:
        missing_tickers = set(expected_tickers) - set(prices.columns)
        if missing_tickers:
            raise ValueError(
                f"Missing expected tickers: {missing_tickers}. " f"Available: {set(prices.columns)}"
            )

    # Check date range (warn if not enough, don't raise error)
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        if prices.index[0] > start_dt:
            logger.warning(
                f"Data starts at {prices.index[0]} but expected >= {start_dt}. "
                "Using available data range."
            )

    if end_date is not None:
        end_dt = pd.to_datetime(end_date)
        if prices.index[-1] < end_dt:
            logger.warning(
                f"Data ends at {prices.index[-1]} but expected <= {end_dt}. "
                "Using available data range."
            )

    # Check for non-positive prices (log warning, not error)
    n_nonpositive = (prices <= 0).sum().sum()
    if n_nonpositive > 0:
        logger.warning(
            f"Found {n_nonpositive} non-positive prices. "
            "These will be set to NaN during preprocessing."
        )

    # Check dtypes (should be numeric)
    non_numeric_cols = [
        col for col in prices.columns if not pd.api.types.is_numeric_dtype(prices[col])
    ]
    if non_numeric_cols:
        raise ValueError(f"Non-numeric columns found: {non_numeric_cols}")

    logger.info(f"Validation passed: {len(prices)} days, {len(prices.columns)} tickers")
