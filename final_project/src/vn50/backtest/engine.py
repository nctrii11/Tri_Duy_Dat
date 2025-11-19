"""Walk-forward backtest engine."""

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.vn50.backtest import metrics

logger = logging.getLogger(__name__)


class BacktestResult:
    """Container for backtest results."""

    def __init__(
        self,
        equity_curves: pd.DataFrame,
        weights_history: dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        metrics_dict: dict[str, dict[str, float]],
    ):
        self.equity_curves = equity_curves
        self.weights_history = weights_history
        self.returns = returns
        self.metrics = metrics_dict


def walk_forward(
    prices: pd.DataFrame,
    optimize_fn: Callable[[pd.DataFrame], dict[str, pd.Series]],
    rebal_freq: str = "M",
    window_in_sample_months: int = 36,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.05,
) -> BacktestResult:
    """
    Walk-forward backtest.

    Args:
        prices: DataFrame with prices (DatetimeIndex)
        optimize_fn: Function that takes returns and returns dict of weights
        rebal_freq: Rebalancing frequency ("M" for monthly, "Q" for quarterly)
        window_in_sample_months: In-sample window size in months
        transaction_cost_bps: Transaction cost in basis points
        slippage_bps: Slippage in basis points
        benchmark_returns: Benchmark returns (optional)
        risk_free_rate: Risk-free rate for Sharpe calculation

    Returns:
        BacktestResult object
    """
    # Convert prices to monthly if needed
    if rebal_freq == "M":
        prices_rebal = prices.resample("M").last()
        freq_days = 21  # Approximate trading days per month
    elif rebal_freq == "Q":
        prices_rebal = prices.resample("Q").last()
        freq_days = 63  # Approximate trading days per quarter
    else:
        raise ValueError(f"Unknown rebal_freq: {rebal_freq}")

    # Calculate returns for optimization
    returns_monthly = prices_rebal.pct_change().dropna()

    # Initialize
    n_periods = len(prices_rebal)
    n_assets = len(prices_rebal.columns)
    equity_curves = {}
    weights_history = {}
    portfolio_returns = []

    # Walk-forward
    for t in range(window_in_sample_months, n_periods):
        # In-sample period
        in_sample_start = max(0, t - window_in_sample_months)
        in_sample_returns = returns_monthly.iloc[in_sample_start:t]

        if len(in_sample_returns) < 12:  # Need at least 12 months
            logger.warning(f"Skipping period {t}: insufficient data")
            continue

        # Optimize
        try:
            weights_dict = optimize_fn(in_sample_returns)
        except Exception as e:
            logger.error(f"Optimization failed at period {t}: {e}")
            continue

        # Use first strategy (e.g., tangency) if multiple
        strategy_name = list(weights_dict.keys())[0]
        weights = weights_dict[strategy_name]

        # Store weights
        if strategy_name not in weights_history:
            weights_history[strategy_name] = pd.DataFrame(
                index=prices_rebal.index, columns=prices_rebal.columns
            )
        weights_history[strategy_name].loc[prices_rebal.index[t]] = weights

        # Calculate portfolio return for this period
        if t < n_periods - 1:
            # Align weights with available prices
            available_tickers = weights.index.intersection(prices_rebal.columns)
            weights_aligned = weights.loc[available_tickers]
            weights_aligned = weights_aligned / weights_aligned.sum()  # Renormalize

            period_return = (
                prices_rebal.loc[prices_rebal.index[t + 1], available_tickers] /
                prices_rebal.loc[prices_rebal.index[t], available_tickers] - 1
            ).dot(weights_aligned)

            # Apply transaction costs and slippage
            total_cost_bps = transaction_cost_bps + slippage_bps
            period_return_net = period_return - total_cost_bps / 10000

            portfolio_returns.append(period_return_net)

            # Update equity curve
            if strategy_name not in equity_curves:
                equity_curves[strategy_name] = [1.0] * (t + 1)
            equity_curves[strategy_name].append(
                equity_curves[strategy_name][-1] * (1 + period_return_net)
            )

    # Convert equity curves to DataFrame
    max_len = max(len(curve) for curve in equity_curves.values()) if equity_curves else 0
    if max_len > 0:
        equity_curves_df = pd.DataFrame(
            {k: v + [v[-1]] * (max_len - len(v)) if len(v) < max_len else v for k, v in equity_curves.items()},
            index=prices_rebal.index[window_in_sample_months : window_in_sample_months + max_len]
        )
    else:
        equity_curves_df = pd.DataFrame()

    # Calculate metrics
    if len(portfolio_returns) > 0:
        portfolio_returns_series = pd.Series(
            portfolio_returns,
            index=prices_rebal.index[window_in_sample_months + 1 : window_in_sample_months + 1 + len(portfolio_returns)]
        )
    else:
        portfolio_returns_series = pd.Series(dtype=float)

    metrics_dict = {}
    for strategy_name in equity_curves.keys():
        strategy_returns = portfolio_returns_series
        ann_return = metrics.calculate_annualized_return(strategy_returns)
        ann_vol = metrics.calculate_annualized_volatility(strategy_returns)
        sharpe = metrics.calculate_sharpe_ratio(strategy_returns, risk_free_rate)
        max_dd = metrics.calculate_max_drawdown(equity_curves_df[strategy_name])
        turnover = metrics.calculate_turnover(weights_history[strategy_name].dropna())

        metrics_dict[strategy_name] = {
            "ann_return": float(ann_return),
            "ann_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "turnover": float(turnover),
        }

        # Tracking error if benchmark provided
        if benchmark_returns is not None:
            tracking_error = metrics.calculate_tracking_error(
                strategy_returns, benchmark_returns
            )
            beta = metrics.calculate_beta(strategy_returns, benchmark_returns)
            metrics_dict[strategy_name]["tracking_error"] = float(tracking_error)
            metrics_dict[strategy_name]["beta"] = float(beta)

    logger.info(f"Backtest completed: {len(portfolio_returns)} periods")

    return BacktestResult(
        equity_curves=equity_curves_df,
        weights_history=weights_history,
        returns=portfolio_returns_series,
        metrics_dict=metrics_dict,
    )

