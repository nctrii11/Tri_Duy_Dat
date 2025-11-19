"""Data loading and preprocessing."""

from . import fetch  # re-export module for backward compatibility
from .fetch import fetch_prices

__all__ = ["fetch", "fetch_prices"]
