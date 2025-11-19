"""Calculate returns from prices."""

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def make_returns(
    prices: pd.DataFrame,
    kind: Literal["log", "arith"] = "log",
    window: int = 1,
    freq: Literal["D", "M"] = "D",
) -> pd.DataFrame:
    """
    Calculate returns from prices.

    Args:
        prices: DataFrame with DatetimeIndex, columns are tickers
        kind: "log" for log returns, "arith" for arithmetic returns
        window: Window size for returns
        freq: "D" for daily, "M" for monthly (resample to end of month)

    Returns:
        DataFrame with returns (first window rows will have NaN, not dropped)
    """
    if freq == "M":
        # Resample to end of month first
        prices_resampled = prices.resample("M").last()
        logger.info(f"Resampled to monthly (end of month): {len(prices_resampled)} months")
    else:
        prices_resampled = prices

    if kind == "log":
        # Log returns: log(p_t / p_{t-window})
        returns = np.log(prices_resampled / prices_resampled.shift(window))
        logger.info(f"Calculated {window}-period log returns")

    elif kind == "arith":
        # Arithmetic returns: (p_t - p_{t-window}) / p_{t-window}
        returns = prices_resampled.pct_change(periods=window)
        logger.info(f"Calculated {window}-period arithmetic returns")

    else:
        raise ValueError(f"Unknown kind: {kind}")

    # Only drop rows where ALL columns are NaN (not first window rows)
    returns = returns.dropna(how="all")

    logger.info(f"Returns: {len(returns)} periods, {len(returns.columns)} tickers")

    return returns
