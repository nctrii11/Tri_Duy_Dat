"""Preprocess price data: cleaning, winsorization, stale-masking."""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_trading_calendar(
    calendar_path: str | Path | None = None,
) -> pd.DatetimeIndex | None:
    """
    Load HOSE trading calendar from CSV file.

    Args:
        calendar_path: Path to calendar CSV file (with 'date' column)

    Returns:
        DatetimeIndex of trading days, or None if file not found
    """
    if calendar_path is None:
        return None

    calendar_path = Path(calendar_path)
    if not calendar_path.exists():
        logger.warning(
            f"Trading calendar file not found: {calendar_path}. " "Skipping calendar alignment."
        )
        return None

    try:
        calendar_df = pd.read_csv(calendar_path)
        if "date" not in calendar_df.columns:
            logger.warning("Calendar file missing 'date' column. " "Skipping calendar alignment.")
            return None

        trading_days = pd.to_datetime(calendar_df["date"]).sort_values()
        # Remove duplicates if any
        trading_days = trading_days.drop_duplicates()
        logger.info(
            f"Loaded {len(trading_days)} trading days from calendar "
            f"({trading_days[0].date()} to {trading_days[-1].date()})"
        )
        return trading_days

    except Exception as e:
        logger.warning(f"Error loading calendar: {e}. Skipping calendar alignment.")
        return None


def clean_prices(
    prices: pd.DataFrame,
    max_missing_ratio: float,
    deduplicate: bool = True,
    align_to_calendar: bool = True,
    calendar_path: str | Path | None = None,
    drop_nonpositive: bool = True,
    remove_all_nan_days: bool = True,
) -> pd.DataFrame:
    """
    Clean price data.

    Args:
        prices: DataFrame with DatetimeIndex, columns are tickers
        max_missing_ratio: Drop ticker if missing ratio > this
        deduplicate: Remove duplicate timestamps
        align_to_calendar: Align to trading calendar
        calendar_path: Path to trading calendar CSV file
        drop_nonpositive: Set non-positive prices to NaN
        remove_all_nan_days: Drop days where all values are NaN
        (No forward-fill as per rules)

    Returns:
        Cleaned DataFrame
    """
    cleaned = prices.copy()

    # Ensure DatetimeIndex
    if not isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned.index = pd.to_datetime(cleaned.index)

    # Sort by date
    cleaned = cleaned.sort_index()

    # Deduplicate timestamps
    if deduplicate:
        cleaned = cleaned[~cleaned.index.duplicated(keep="first")]
        cleaned = cleaned.sort_index()

    # Calendar alignment
    n_days_before_align = len(cleaned)
    if align_to_calendar:
        trading_days = load_trading_calendar(calendar_path)
        if trading_days is not None:
            # Filter trading_days to date range of cleaned data
            trading_days_filtered = trading_days[
                (trading_days >= cleaned.index[0]) & (trading_days <= cleaned.index[-1])
            ]
            cleaned = cleaned.reindex(trading_days_filtered)
            n_days_after_align = len(cleaned)
            n_days_removed = n_days_before_align - n_days_after_align
            logger.info(
                f"Aligned to trading calendar: "
                f"{n_days_before_align} → {n_days_after_align} days "
                f"({n_days_removed} non-trading days removed)"
            )
        else:
            logger.warning(
                "Calendar alignment requested but calendar not loaded. " "Skipping alignment."
            )

    # Convert to float64
    cleaned = cleaned.astype(float)

    # Handle non-positive prices (set to NaN, not drop rows)
    if drop_nonpositive:
        n_nonpositive = (cleaned <= 0).sum().sum()
        if n_nonpositive > 0:
            logger.warning(f"Found {n_nonpositive} non-positive prices, setting to NaN")
        cleaned = cleaned.where(cleaned > 0, np.nan)

    # Filter columns by missing ratio
    missing_ratio = cleaned.isna().sum() / len(cleaned)
    valid_tickers = missing_ratio[missing_ratio <= max_missing_ratio].index.tolist()
    dropped_tickers = set(cleaned.columns) - set(valid_tickers)
    if dropped_tickers:
        logger.warning(f"Dropping tickers with high missing ratio: {dropped_tickers}")
    cleaned = cleaned[valid_tickers]

    # Drop days where all values are NaN
    if remove_all_nan_days:
        all_nan_mask = cleaned.isna().all(axis=1)
        if all_nan_mask.any():
            logger.warning(f"Dropping {all_nan_mask.sum()} days with all NaN")
            cleaned = cleaned[~all_nan_mask]

    logger.info(f"Cleaned prices: {len(cleaned)} days, {len(cleaned.columns)} tickers")

    return cleaned


def winsorize_returns(
    returns: pd.DataFrame,
    mode: Literal["rolling_quantile", "rolling_mad", "abs_cap"] = "rolling_quantile",
    window: int = 252,
    lower_q: float = 0.005,
    upper_q: float = 0.995,
    cap_by_exchange_limit: bool = False,
    abs_cap: float | None = None,
) -> pd.DataFrame:
    """
    Winsorize outliers in returns (no look-ahead).

    Args:
        returns: DataFrame with returns
        mode: "rolling_quantile", "rolling_mad", or "abs_cap"
        window: Rolling window size (for rolling modes)
        lower_q: Lower quantile (for rolling_quantile)
        upper_q: Upper quantile (for rolling_quantile)
        cap_by_exchange_limit: Apply absolute cap after rolling winsorization
        abs_cap: Absolute cap value
            (for abs_cap mode or cap_by_exchange_limit)

    Returns:
        Winsorized DataFrame
    """
    winsorized = returns.copy()

    # Calculate min_periods
    min_periods = max(1, min(25, window // 5))

    if mode == "rolling_quantile":
        # Rolling quantile without look-ahead (shift by 1)
        for col in winsorized.columns:
            # Calculate quantiles on shifted data (no look-ahead)
            q_low = (
                winsorized[col]
                .shift(1)
                .rolling(window=window, min_periods=min_periods)
                .quantile(lower_q)
            )
            q_high = (
                winsorized[col]
                .shift(1)
                .rolling(window=window, min_periods=min_periods)
                .quantile(upper_q)
            )
            winsorized[col] = winsorized[col].clip(lower=q_low, upper=q_high)

        logger.info(
            f"Winsorized using rolling quantile "
            f"(window={window}, quantiles=[{lower_q}, {upper_q}])"
        )

    elif mode == "rolling_mad":
        # Rolling MAD winsorization
        for col in winsorized.columns:
            shifted = winsorized[col].shift(1)
            rolling_median = shifted.rolling(window=window, min_periods=min_periods).median()
            rolling_mad = shifted.rolling(window=window, min_periods=min_periods).apply(
                lambda x: np.median(np.abs(x - np.median(x)))
            )
            lower_bound = rolling_median - 5 * rolling_mad
            upper_bound = rolling_median + 5 * rolling_mad
            winsorized[col] = winsorized[col].clip(lower=lower_bound, upper=upper_bound)

        logger.info(f"Winsorized using rolling MAD (window={window})")

    elif mode == "abs_cap":
        if abs_cap is None:
            raise ValueError("abs_cap must be provided for abs_cap mode")
        winsorized = winsorized.clip(lower=-abs_cap, upper=abs_cap)
        logger.info(f"Winsorized using absolute cap: {abs_cap}")

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Apply exchange limit cap if requested
    if cap_by_exchange_limit:
        if abs_cap is None:
            raise ValueError("abs_cap must be provided when cap_by_exchange_limit=True")
        winsorized = winsorized.clip(lower=-abs_cap, upper=abs_cap)
        logger.info(f"Applied exchange limit cap: ±{abs_cap}")

    return winsorized


def _consecutive_counts_zero(series: pd.Series) -> pd.Series:
    """
    Count consecutive zeros in a series.

    Args:
        series: Series with returns

    Returns:
        Series with counts of consecutive zeros at each position
    """
    zero_mask = (series == 0) | series.isna()
    groups = (zero_mask != zero_mask.shift()).cumsum()
    run_lengths = zero_mask.groupby(groups).transform("sum")
    return run_lengths


def mask_stale_returns(
    returns: pd.DataFrame,
    min_consecutive_days: int = 3,
    warn_threshold_pct: float = 10.0,
) -> pd.DataFrame:
    """
    Mask stale returns (zero return for consecutive days).

    Args:
        returns: DataFrame with returns
        min_consecutive_days: Minimum consecutive days with zero return to mask
        warn_threshold_pct: Warn if ticker has stale returns > this %

    Returns:
        DataFrame with stale returns masked (NaN)
    """
    masked = returns.copy()
    stale_stats = {}  # Track stats per ticker

    # Find consecutive zero returns
    for col in returns.columns:
        zero_counts = _consecutive_counts_zero(returns[col])
        zero_mask = (returns[col] == 0) | returns[col].isna()

        # Mask from min_consecutive_days onwards
        stale_mask = (zero_counts >= min_consecutive_days) & zero_mask
        n_masked = stale_mask.sum()
        if n_masked > 0:
            masked.loc[stale_mask, col] = np.nan
            pct_masked = (n_masked / len(returns)) * 100
            stale_stats[col] = {"n_masked": n_masked, "pct_masked": pct_masked}

            if pct_masked > warn_threshold_pct:
                logger.warning(
                    f"{col}: {n_masked} stale returns "
                    f"({pct_masked:.2f}% of total) - "
                    f"consider investigating thin trading or suspension"
                )
            else:
                logger.info(
                    f"Masked {n_masked} stale returns for {col} "
                    f"(>= {min_consecutive_days} consecutive zero returns, "
                    f"{pct_masked:.2f}%)"
                )

    # Log top tickers with most stale returns
    if stale_stats:
        sorted_stats = sorted(stale_stats.items(), key=lambda x: x[1]["n_masked"], reverse=True)
        top_n = min(5, len(sorted_stats))
        top_tickers_str = ", ".join(
            [
                f"{ticker} ({stats['n_masked']} returns, " f"{stats['pct_masked']:.2f}%)"
                for ticker, stats in sorted_stats[:top_n]
            ]
        )
        logger.info(f"Top {top_n} tickers by stale returns: {top_tickers_str}")

    return masked
