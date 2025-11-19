"""Unified CLI to run Markowitz optimization and walk-forward backtest."""

import json
import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.vn50.backtest import walkforward
from src.vn50.optimize import markowitz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_monthly_returns(cfg: DictConfig) -> pd.DataFrame:
    """Load monthly simple returns (prefer Parquet)."""
    processed_dir = Path(cfg.paths.data.processed)
    parquet_path = processed_dir / "returns_monthly_simple.parquet"
    csv_path = processed_dir / "returns_monthly_simple.csv"

    if parquet_path.exists():
        logger.info("Loading monthly returns from %s", parquet_path)
        returns = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        logger.info("Loading monthly returns from %s", csv_path)
        returns = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        raise FileNotFoundError(
            f"returns_monthly_simple file not found in {processed_dir}. "
            "Run preprocess pipeline first."
        )

    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)

    return returns.sort_index()


def _markowitz_solver(mu: pd.Series, sigma: pd.DataFrame, cfg: DictConfig) -> dict:
    """Wrapper around mean_variance_weights returning GMV & Tangency weights."""
    weights = markowitz.mean_variance_weights(
        mu,
        sigma,
        objective="frontier",
        risk_free_rate=cfg.markowitz_manual.risk_free_rate,
        min_weight=cfg.markowitz_manual.min_weight,
        max_weight=cfg.markowitz_manual.max_weight,
        solver=cfg.markowitz_manual.solver,
        max_iter=cfg.markowitz_manual.max_iter,
    )
    return {"gmv": weights.get("gmv"), "tangency": weights.get("tangency")}


def _save_backtest_artifacts(
    cfg: DictConfig,
    backtest_result: walkforward.BacktestResult,
) -> None:
    """Persist metrics, returns, and NAV plot."""
    artifacts_dir = Path(cfg.paths.reports.artifacts)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # NAV plot
    nav_path = artifacts_dir / "nav_walkforward_markowitz.png"
    walkforward.plot_nav_curves(backtest_result, nav_path)

    # Returns CSV
    returns_df = pd.DataFrame(
        {
            name: strat.returns
            for name, strat in backtest_result.strategies.items()
        }
    )
    returns_csv_path = artifacts_dir / "returns_walkforward_markowitz.csv"
    returns_df.to_csv(returns_csv_path, index_label="date")
    logger.info("Saved strategy returns to %s", returns_csv_path)

    # NAV CSV (optional but useful)
    nav_df = pd.DataFrame(
        {name: strat.nav for name, strat in backtest_result.strategies.items()}
    )
    nav_csv_path = artifacts_dir / "nav_walkforward_markowitz.csv"
    nav_df.to_csv(nav_csv_path, index_label="date")
    logger.info("Saved NAV series to %s", nav_csv_path)

    # Metrics log
    metrics = {
        name: strat.metrics for name, strat in backtest_result.strategies.items()
    }
    log_payload = {
        "experiment": cfg.experiment,
        "window_months": cfg.backtest.walkforward.window_months,
        "risk_free_rate": cfg.backtest.rf,
        "n_strategies": len(backtest_result.strategies),
        "metrics": metrics,
    }

    logs_dir = Path(cfg.paths.reports.logs)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "metrics_walkforward_markowitz.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=2)
    logger.info("Saved metrics log to %s", log_path)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run Markowitz optimization and walk-forward backtest end-to-end."""
    logger.info(
        "Running unified Markowitz pipeline for experiment=%s", cfg.experiment
    )

    returns_monthly = _load_monthly_returns(cfg)

    backtest_result = walkforward.walk_forward_backtest(
        returns_monthly=returns_monthly,
        cfg=cfg,
        markowitz_solver_fn=lambda mu, sigma, _: _markowitz_solver(mu, sigma, cfg),
    )

    if not backtest_result.strategies:
        raise RuntimeError("Walk-forward backtest produced no strategy results.")

    _save_backtest_artifacts(cfg, backtest_result)
    logger.info("Completed Markowitz run successfully.")


if __name__ == "__main__":
    main()

