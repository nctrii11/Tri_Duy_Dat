"""Walk-forward backtest for Markowitz portfolio optimization."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    """Result for a single strategy."""

    name: str
    nav: pd.Series  # chuỗi giá trị danh mục (bắt đầu từ 1.0)
    returns: pd.Series  # chuỗi monthly returns của chiến lược
    weights: dict  # mapping: rebalance_date -> pd.Series(weights)
    metrics: dict  # ann_return, ann_vol, sharpe, max_dd, turnover, v.v.


@dataclass
class BacktestResult:
    """Container for walk-forward backtest results."""

    strategies: dict  # {"gmv": StrategyResult, "tangency": ..., "equal_weight": ...}
    benchmark: pd.Series | None = None  # nếu có thêm VNINDEX / VN30 index


def compute_metrics(
    strategy_returns: pd.Series,
    nav: pd.Series,
    rf_annual: float = 0.03,
) -> dict:
    """
    Tính các metrics cơ bản.

    Args:
        strategy_returns: Monthly simple returns (pd.Series)
        nav: NAV series (pd.Series)
        rf_annual: Annualized risk-free rate (default: 0.03)

    Returns:
        Dictionary với các metrics:
          - ann_return: Annualized return
          - ann_vol: Annualized volatility
          - sharpe: Sharpe ratio
          - max_drawdown: Maximum drawdown
          - calmar: Calmar ratio (optional)
          - sortino: Sortino ratio (optional)
    """
    if len(strategy_returns) == 0:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
        }

    # Annualized return: (1 + r_monthly).prod()^(12/n) - 1
    n_months = len(strategy_returns)
    if n_months > 0:
        total_return = (1 + strategy_returns).prod()
        ann_return = total_return ** (12 / n_months) - 1
    else:
        ann_return = np.nan

    # Annualized volatility: std_monthly * sqrt(12)
    ann_vol = strategy_returns.std() * np.sqrt(12)

    # Sharpe ratio: (ann_return - rf) / ann_vol
    sharpe = (ann_return - rf_annual) / ann_vol if ann_vol > 0 else np.nan

    # Maximum drawdown
    cummax = nav.expanding().max()
    drawdown = (nav - cummax) / cummax
    max_drawdown = drawdown.min()

    # Calmar ratio: ann_return / abs(max_drawdown)
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    # Sortino ratio (downside deviation)
    downside_returns = strategy_returns[strategy_returns < 0]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std() * np.sqrt(12)
        sortino = (ann_return - rf_annual) / downside_std if downside_std > 0 else np.nan
    else:
        sortino = np.nan

    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "calmar": float(calmar),
        "sortino": float(sortino),
    }


def compute_turnover(weights_history: dict) -> float:
    """
    Tính turnover trung bình năm.

    Args:
        weights_history: dict[rebalance_date -> pd.Series(weights)]

    Returns:
        Turnover trung bình năm (annualized)
    """
    if len(weights_history) < 2:
        return 0.0

    # Sắp xếp theo date
    dates = sorted(weights_history.keys())
    turnovers = []

    for i in range(1, len(dates)):
        w_prev = weights_history[dates[i - 1]]
        w_curr = weights_history[dates[i]]

        # Align tickers
        common_tickers = w_prev.index.intersection(w_curr.index)
        if len(common_tickers) == 0:
            continue

        w_prev_aligned = w_prev.loc[common_tickers]
        w_curr_aligned = w_curr.loc[common_tickers]

        # Turnover = 0.5 * sum(|w_t - w_{t-1}|)
        turnover = 0.5 * (w_curr_aligned - w_prev_aligned).abs().sum()
        turnovers.append(turnover)

    if len(turnovers) == 0:
        return 0.0

    # Annualize: nhân với số lần tái cân bằng mỗi năm (12 nếu monthly)
    avg_turnover_per_rebal = np.mean(turnovers)
    annual_turnover = avg_turnover_per_rebal * 12  # Assuming monthly rebalancing

    return float(annual_turnover)


def walk_forward_backtest(
    returns_monthly: pd.DataFrame,
    cfg,
    markowitz_solver_fn: Callable,
) -> BacktestResult:
    """
    Walk-forward backtest cho Markowitz portfolio.

    Args:
        returns_monthly: DataFrame (index: month-end dates, columns: tickers)
        cfg: Hydra config, dùng:
            cfg.backtest.walkforward.window_months (vd: 36)
            cfg.backtest.walkforward.rebalance_freq (M)
            cfg.backtest.rf
            cfg.markowitz_manual.bounds.min_weight, max_weight, v.v.
        markowitz_solver_fn: Hàm gọi xuống optimizer Markowitz manual,
            ví dụ: def markowitz_solver_fn(mu, Sigma, cfg) -> dict:
                -> {'gmv': Series, 'tangency': Series, ...}

    Returns:
        BacktestResult với strategies: GMV, Tangency, Equal-weight
    """
    from src.vn50.markowitz import estimators

    # 1) Sắp xếp index, đảm bảo là DatetimeIndex cuối tháng
    returns_monthly = returns_monthly.sort_index()
    if not isinstance(returns_monthly.index, pd.DatetimeIndex):
        returns_monthly.index = pd.to_datetime(returns_monthly.index)

    # 2) Xác định các mốc tái cân bằng monthly
    rebalance_dates = returns_monthly.index.tolist()
    window_months = cfg.backtest.walkforward.window_months
    rf_annual = cfg.backtest.rf

    logger.info(
        f"Starting walk-forward backtest: "
        f"window={window_months} months, "
        f"n_periods={len(rebalance_dates)}, "
        f"n_assets={len(returns_monthly.columns)}"
    )

    # 3) Khởi tạo storage cho weights và returns
    weights_gmv = {}
    weights_tangency = {}
    weights_equal = {}
    returns_gmv = []
    returns_tangency = []
    returns_equal = []
    returns_dates = []

    # Equal-weight weights (cố định)
    n_assets = len(returns_monthly.columns)
    w_equal_fixed = pd.Series(1.0 / n_assets, index=returns_monthly.columns)

    # 4) Walk-forward loop
    n_train_periods = 0
    n_test_periods = 0

    for t_idx, rebalance_date in enumerate(rebalance_dates):
        # Kiểm tra đủ lịch sử
        if t_idx < window_months:
            continue

        # Xác định train_window = [t - window_months, t-1]
        train_start_idx = t_idx - window_months
        train_end_idx = t_idx - 1  # Không dùng data tại t_idx (no look-ahead)
        train_window = rebalance_dates[train_start_idx : train_end_idx + 1]

        # Lấy returns_train
        returns_train = returns_monthly.loc[train_window]

        # Loại bỏ NaN
        returns_train = returns_train.dropna(axis=0, how="all").dropna(axis=1, how="any")

        if len(returns_train) < 12:  # Cần ít nhất 12 tháng
            logger.warning(f"Insufficient data at {rebalance_date}, skipping")
            continue

        n_train_periods += 1

        # Ước lượng mu, Sigma từ returns_train
        try:
            mu, Sigma = estimators.estimate_mu_sigma(
                returns_train,
                shrinkage=cfg.features.cov.shrinkage,
                shrinkage_factor=cfg.features.cov.shrinkage_factor,
                jitter=cfg.features.cov.jitter,
                annualization_factor=12,
            )
        except Exception as e:
            logger.error(f"Failed to estimate mu/Sigma at {rebalance_date}: {e}")
            continue

        # Gọi markowitz_solver_fn để lấy trọng số
        try:
            weights_dict = markowitz_solver_fn(mu, Sigma, cfg)
        except Exception as e:
            logger.error(f"Optimization failed at {rebalance_date}: {e}")
            continue

        # Lấy weights
        w_gmv = weights_dict.get("gmv")
        w_tangency = weights_dict.get("tangency")

        if w_gmv is None or w_tangency is None:
            logger.warning(f"Missing weights at {rebalance_date}, skipping")
            continue

        # Lưu weights tại rebalance_date (weights này được tính từ data đến t-1)
        weights_gmv[rebalance_date] = w_gmv
        weights_tangency[rebalance_date] = w_tangency
        weights_equal[rebalance_date] = w_equal_fixed

        # Tính performance ở kỳ tiếp theo (period t+1) - dùng returns tại period tiếp theo
        # Đảm bảo không có look-ahead: weights tại t được áp dụng cho returns tại t+1
        if t_idx < len(rebalance_dates) - 1:
            next_date = rebalance_dates[t_idx + 1]
            returns_next = returns_monthly.loc[next_date]

            # Align tickers
            tickers_gmv = w_gmv.index.intersection(returns_next.index)
            tickers_tangency = w_tangency.index.intersection(returns_next.index)
            tickers_equal = w_equal_fixed.index.intersection(returns_next.index)

            if len(tickers_gmv) > 0:
                w_gmv_aligned = w_gmv.loc[tickers_gmv]
                w_gmv_aligned = w_gmv_aligned / w_gmv_aligned.sum()  # Renormalize
                r_gmv_t = (returns_next.loc[tickers_gmv] * w_gmv_aligned).sum()
                returns_gmv.append(r_gmv_t)
            else:
                returns_gmv.append(0.0)

            if len(tickers_tangency) > 0:
                w_tangency_aligned = w_tangency.loc[tickers_tangency]
                w_tangency_aligned = w_tangency_aligned / w_tangency_aligned.sum()  # Renormalize
                r_tangency_t = (returns_next.loc[tickers_tangency] * w_tangency_aligned).sum()
                returns_tangency.append(r_tangency_t)
            else:
                returns_tangency.append(0.0)

            if len(tickers_equal) > 0:
                r_equal_t = returns_next.loc[tickers_equal].mean()
                returns_equal.append(r_equal_t)
            else:
                returns_equal.append(0.0)

            returns_dates.append(next_date)
            n_test_periods += 1

    logger.info(
        f"Walk-forward completed: "
        f"n_train_periods={n_train_periods}, "
        f"n_test_periods={n_test_periods}"
    )

    # 5) Tính NAV và metrics cho từng chiến lược
    strategies = {}

    # GMV
    if len(returns_gmv) > 0:
        returns_gmv_series = pd.Series(returns_gmv, index=returns_dates)
        # NAV bắt đầu từ 1.0, sau đó nhân với (1 + return) mỗi kỳ
        nav_gmv = (1 + returns_gmv_series).cumprod()
        # Thêm giá trị ban đầu 1.0 tại ngày đầu tiên
        first_date = returns_dates[0] if len(returns_dates) > 0 else rebalance_dates[window_months]
        nav_gmv = pd.Series([1.0] + nav_gmv.tolist(), index=[first_date] + returns_dates)

        metrics_gmv = compute_metrics(returns_gmv_series, nav_gmv, rf_annual)
        metrics_gmv["turnover"] = compute_turnover(weights_gmv)

        strategies["gmv"] = StrategyResult(
            name="gmv",
            nav=nav_gmv,
            returns=returns_gmv_series,
            weights=weights_gmv,
            metrics=metrics_gmv,
        )

    # Tangency
    if len(returns_tangency) > 0:
        returns_tangency_series = pd.Series(returns_tangency, index=returns_dates)
        nav_tangency = (1 + returns_tangency_series).cumprod()
        first_date = returns_dates[0] if len(returns_dates) > 0 else rebalance_dates[window_months]
        nav_tangency = pd.Series(
            [1.0] + nav_tangency.tolist(),
            index=[first_date] + returns_dates,
        )

        metrics_tangency = compute_metrics(returns_tangency_series, nav_tangency, rf_annual)
        metrics_tangency["turnover"] = compute_turnover(weights_tangency)

        strategies["tangency"] = StrategyResult(
            name="tangency",
            nav=nav_tangency,
            returns=returns_tangency_series,
            weights=weights_tangency,
            metrics=metrics_tangency,
        )

    # Equal-weight
    if len(returns_equal) > 0:
        returns_equal_series = pd.Series(returns_equal, index=returns_dates)
        nav_equal = (1 + returns_equal_series).cumprod()
        first_date = returns_dates[0] if len(returns_dates) > 0 else rebalance_dates[window_months]
        nav_equal = pd.Series(
            [1.0] + nav_equal.tolist(),
            index=[first_date] + returns_dates,
        )

        metrics_equal = compute_metrics(returns_equal_series, nav_equal, rf_annual)
        metrics_equal["turnover"] = compute_turnover(weights_equal)

        strategies["equal_weight"] = StrategyResult(
            name="equal_weight",
            nav=nav_equal,
            returns=returns_equal_series,
            weights=weights_equal,
            metrics=metrics_equal,
        )

    return BacktestResult(strategies=strategies, benchmark=None)


def plot_nav_curves(backtest_result: BacktestResult, output_path: str | Path | None = None) -> None:
    """
    Vẽ biểu đồ NAV cho các chiến lược.

    Args:
        backtest_result: BacktestResult object
        output_path: Đường dẫn lưu file (optional)
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    for name, strat in backtest_result.strategies.items():
        plt.plot(strat.nav.index, strat.nav.values, label=name.upper(), linewidth=2)

    plt.xlabel("Date", fontsize=12)
    plt.ylabel("NAV (start = 1.0)", fontsize=12)
    plt.title("Walk-forward Backtest – Markowitz vs Equal-weight (VN30)", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved NAV plot to {output_path}")
    else:
        plt.show()

    plt.close()

