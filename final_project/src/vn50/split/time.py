"""Time-based data splitting for in-sample/out-of-sample."""

import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


def split_time(
    data: pd.DataFrame,
    in_sample_years: int = 3,
    out_of_sample_years: int = 2,
    internal_val_split: float | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Split data into in-sample and out-of-sample periods.

    Args:
        data: DataFrame with DatetimeIndex
        in_sample_years: Number of years for in-sample
        out_of_sample_years: Number of years for out-of-sample
        internal_val_split: Optional fraction for internal validation

    Returns:
        Dictionary with keys: "in_sample", "out_of_sample", optionally "train", "val"
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be DatetimeIndex")

    sorted_data = data.sort_index()
    start_date = sorted_data.index[0]
    end_date = sorted_data.index[-1]

    # Calculate split dates
    in_sample_end = start_date + pd.DateOffset(years=in_sample_years)
    out_of_sample_start = in_sample_end
    out_of_sample_end = out_of_sample_start + pd.DateOffset(years=out_of_sample_years)

    # Ensure we don't exceed data range
    if out_of_sample_end > end_date:
        logger.warning(
            f"Out-of-sample end date {out_of_sample_end} exceeds data end {end_date}. "
            f"Using {end_date} instead."
        )
        out_of_sample_end = end_date

    # Split data
    in_sample = sorted_data.loc[start_date:in_sample_end]
    out_of_sample = sorted_data.loc[out_of_sample_start:out_of_sample_end]

    result = {
        "in_sample": in_sample,
        "out_of_sample": out_of_sample,
    }

    # Optional internal train/val split
    if internal_val_split is not None:
        val_start = start_date + pd.DateOffset(
            years=in_sample_years * (1 - internal_val_split)
        )
        train = sorted_data.loc[start_date:val_start]
        val = sorted_data.loc[val_start:in_sample_end]
        result["train"] = train
        result["val"] = val

    logger.info(
        f"Time split: in_sample={len(in_sample)} periods "
        f"({start_date.date()} to {in_sample_end.date()}), "
        f"out_of_sample={len(out_of_sample)} periods "
        f"({out_of_sample_start.date()} to {out_of_sample_end.date()})"
    )

    return result

