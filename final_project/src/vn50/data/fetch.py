"""Data được cào lại từ vnstock cho VN30 giai đoạn 2020-10-30 đến 2025-10-30 để phục vụ cho Markowitz backtest."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from vnstock import Quote
except ImportError as exc:  # pragma: no cover - guarded by dependency management
    raise ImportError(
        "The vnstock package is required to fetch prices. "
        "Install it via `uv add vnstock` or `uv sync`."
    ) from exc


_DATE_COLUMNS: tuple[str, ...] = ("date", "time", "trading_date", "tradingDate")
_ADJUSTED_PRICE_COLUMNS: tuple[str, ...] = (
    "adjust_price",
    "adj_close",
    "close_adjusted",
    "average_price_adjusted",
)
_RAW_PRICE_COLUMNS: tuple[str, ...] = (
    "close",
    "close_price",
    "last",
    "match_price",
)


def _resolve_data_source(source: str | None) -> str:
    default_source = "vci"
    if source is None:
        return default_source
    normalized = source.lower()
    if normalized == "vnstock":
        return default_source
    return normalized


def _download_symbol_history(symbol: str, start: str, end: str, source: str) -> pd.DataFrame:
    quote = Quote(symbol=symbol, source=source)
    data = quote.history(start=start, end=end, interval="1D")
    if data is None or data.empty:
        raise ValueError(f"vnstock returned no data for {symbol}")
    return data


def _pick_date_column(raw: pd.DataFrame) -> pd.Series:
    available = {col.lower(): col for col in raw.columns}
    for candidate in _DATE_COLUMNS:
        if candidate.lower() in available:
            return pd.to_datetime(raw[available[candidate.lower()]], errors="coerce")
    raise ValueError("No date-like column found in vnstock payload")


def _pick_price_column(raw: pd.DataFrame, adjusted: bool) -> str:
    normalized = {col.lower(): col for col in raw.columns}
    ordered_candidates: list[str] = []
    if adjusted:
        ordered_candidates.extend(_ADJUSTED_PRICE_COLUMNS)
    ordered_candidates.extend(_RAW_PRICE_COLUMNS)
    for candidate in ordered_candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise ValueError(
        f"Could not locate a price column (adjusted={adjusted}) in payload columns {list(raw.columns)}"
    )


def _series_from_payload(raw: pd.DataFrame, symbol: str, adjusted: bool) -> pd.Series:
    date_index = _pick_date_column(raw)
    price_column = _pick_price_column(raw, adjusted)
    series = pd.Series(raw[price_column].to_numpy(dtype=float), index=date_index, name=symbol)
    series = series[~series.index.isna()]
    series.index = series.index.tz_localize(None)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series = series.dropna()
    if series.empty:
        raise ValueError(f"No valid price observations for {symbol}")
    return series


def _intersect_indexes(series_list: Iterable[pd.Series]) -> pd.DatetimeIndex:
    iterator = iter(series_list)
    try:
        first_series = next(iterator)
    except StopIteration as exc:
        raise ValueError("No price series available to align") from exc

    common_index = first_series.index
    for series in iterator:
        common_index = common_index.intersection(series.index)
        if common_index.empty:
            raise ValueError(
                "Fetched data has no overlapping trading days across symbols. "
                "Please verify the requested date range."
            )
    return common_index.sort_values()


def fetch_prices(
    symbols: list[str],
    start: str,
    end: str,
    adjusted: bool = True,
    source: str = "vnstock",
    max_retries: int = 3,
    sleep_seconds: float = 0.5,
) -> pd.DataFrame:
    """
    Fetch daily prices for VN30 symbols from vnstock within the requested window.

    Returns a wide DataFrame (DatetimeIndex, columns=symbols) containing adjusted close
    when available, otherwise the close price, aligned on common trading days only.
    """
    if not symbols:
        raise ValueError("Symbol universe cannot be empty")
    unique_symbols = list(dict.fromkeys([symbol.upper() for symbol in symbols]))

    aligned_payloads: dict[str, pd.Series] = {}
    failed_symbols: dict[str, str] = {}
    resolved_source = _resolve_data_source(source)

    for symbol in unique_symbols:
        logger.info("Fetching %s from vnstock (%s → %s)", symbol, start, end)
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                raw_frame = _download_symbol_history(symbol, start, end, resolved_source)
                price_series = _series_from_payload(raw_frame, symbol, adjusted)
                aligned_payloads[symbol] = price_series
                logger.info("Fetched %s observations for %s", len(price_series), symbol)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Attempt %s/%s failed for %s: %s", attempt, max_retries, symbol, exc
                )
                if attempt < max_retries:
                    time.sleep(sleep_seconds)
        if last_error is not None:
            failed_symbols[symbol] = str(last_error)

    if failed_symbols:
        error_messages = ", ".join(f"{sym}: {msg}" for sym, msg in failed_symbols.items())
        raise RuntimeError(f"Failed to fetch symbols: {error_messages}")

    common_index = _intersect_indexes(aligned_payloads.values())
    logger.info("Common trading days after alignment: %s", len(common_index))

    prices = pd.DataFrame(
        {symbol: series.reindex(common_index) for symbol, series in aligned_payloads.items()}
    )
    prices.index.name = "date"
    return prices
