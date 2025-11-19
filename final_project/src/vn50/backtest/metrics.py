"""Backtest metrics: returns, volatility, Sharpe, drawdown, turnover, tracking error."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_annualized_return(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Calculate annualized return."""
    if isinstance(returns, pd.Series):
        return (1 + returns).prod() ** (252 / len(returns)) - 1
    else:
        return returns.apply(lambda col: (1 + col).prod() ** (252 / len(col)) - 1)


def calculate_annualized_volatility(returns: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Calculate annualized volatility."""
    if isinstance(returns, pd.Series):
        return returns.std() * np.sqrt(252)
    else:
        return returns.std() * np.sqrt(252)


def calculate_sharpe_ratio(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: float = 0.05,
) -> float | pd.Series:
    """Calculate Sharpe ratio (annualized)."""
    ann_return = calculate_annualized_return(returns)
    ann_vol = calculate_annualized_volatility(returns)

    if isinstance(returns, pd.Series):
        return (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan
    else:
        return (ann_return - risk_free_rate) / ann_vol


def calculate_max_drawdown(equity_curve: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Calculate maximum drawdown."""
    if isinstance(equity_curve, pd.Series):
        cummax = equity_curve.expanding().max()
        drawdown = (equity_curve - cummax) / cummax
        return drawdown.min()
    else:
        return equity_curve.apply(
            lambda col: ((col - col.expanding().max()) / col.expanding().max()).min()
        )


def calculate_turnover(weights: pd.DataFrame) -> float:
    """
    Calculate average turnover.

    Args:
        weights: DataFrame with weights over time (rows=dates, cols=tickers)

    Returns:
        Average turnover (fraction)
    """
    weight_changes = weights.diff().abs().sum(axis=1)
    return weight_changes.mean()


def calculate_tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Calculate tracking error (annualized).

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Annualized tracking error
    """
    # Align by date
    aligned = pd.DataFrame({
        "portfolio": portfolio_returns,
        "benchmark": benchmark_returns,
    }).dropna()

    if len(aligned) == 0:
        return np.nan

    active_returns = aligned["portfolio"] - aligned["benchmark"]
    return active_returns.std() * np.sqrt(252)


def calculate_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Calculate beta vs benchmark.

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Beta
    """
    # Align by date
    aligned = pd.DataFrame({
        "portfolio": portfolio_returns,
        "benchmark": benchmark_returns,
    }).dropna()

    if len(aligned) < 2:
        return np.nan

    covariance = aligned["portfolio"].cov(aligned["benchmark"])
    benchmark_var = aligned["benchmark"].var()

    if benchmark_var == 0:
        return np.nan

    return covariance / benchmark_var

