"""CLI: Run backtest."""

import json
import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.vn50.backtest import engine, metrics
from src.vn50.markowitz import estimators
from src.vn50.optimize import markowitz
from src.vn50.split import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run backtest."""
    logger.info(f"Step 6: Backtest for experiment: {cfg.experiment}")

    # Load prices and returns
    prices_path = Path(cfg.paths.data.processed) / "prices_clean.csv"
    returns_path = Path(cfg.paths.data.processed) / "returns_monthly.csv"

    if not prices_path.exists() or not returns_path.exists():
        raise FileNotFoundError("Required data files not found")

    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)

    # Split time
    splits = time.split_time(
        returns,
        in_sample_years=cfg.split.in_sample_years,
        out_of_sample_years=cfg.split.out_of_sample_years,
    )

    out_of_sample_returns = splits["out_of_sample"]
    out_of_sample_prices = prices.loc[out_of_sample_returns.index[0] : out_of_sample_returns.index[-1]]

    # Define optimization function for manual
    def optimize_manual(returns_in_sample: pd.DataFrame) -> dict[str, pd.Series]:
        mu, Sigma = estimators.estimate_mu_sigma(
            returns_in_sample,
            shrinkage=cfg.features.cov.shrinkage,
            shrinkage_factor=cfg.features.cov.shrinkage_factor,
            jitter=cfg.features.cov.jitter,
            annualization_factor=12,
        )
        weights_dict = markowitz.mean_variance_weights(
            mu,
            Sigma,
            objective="tangency",
            risk_free_rate=cfg.markowitz_manual.risk_free_rate,
            min_weight=cfg.markowitz_manual.min_weight,
            max_weight=cfg.markowitz_manual.max_weight,
            solver=cfg.markowitz_manual.solver,
            max_iter=cfg.markowitz_manual.max_iter,
        )
        return weights_dict

    # Define optimization function for PyPortfolioOpt
    def optimize_pypfopt(returns_in_sample: pd.DataFrame) -> dict[str, pd.Series]:
        weights_dict = markowitz.optimize_markowitz_pypfopt(
            returns_in_sample,
            risk_model=cfg.markowitz_pypfopt.risk_model,
            objective="max_sharpe",
            risk_free_rate=cfg.markowitz_manual.risk_free_rate,
            weight_bounds=(
                cfg.markowitz_pypfopt.weight_bounds.min,
                cfg.markowitz_pypfopt.weight_bounds.max,
            ),
        )
        return weights_dict

    # Run backtests
    result_manual = engine.walk_forward(
        out_of_sample_prices,
        optimize_manual,
        rebal_freq=cfg.backtest.rebal_freq,
        window_in_sample_months=cfg.backtest.window_in_sample_months,
        transaction_cost_bps=cfg.backtest.transaction_cost_bps,
        slippage_bps=cfg.backtest.slippage_bps,
        risk_free_rate=cfg.markowitz_manual.risk_free_rate,
    )

    result_pypfopt = engine.walk_forward(
        out_of_sample_prices,
        optimize_pypfopt,
        rebal_freq=cfg.backtest.rebal_freq,
        window_in_sample_months=cfg.backtest.window_in_sample_months,
        transaction_cost_bps=cfg.backtest.transaction_cost_bps,
        slippage_bps=cfg.backtest.slippage_bps,
        risk_free_rate=cfg.markowitz_manual.risk_free_rate,
    )

    # Save results
    output_dir = Path(cfg.paths.reports.artifacts)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Equity curves
    result_manual.equity_curves.to_csv(
        output_dir / f"equity_curve_manual_{cfg.experiment}.csv"
    )
    result_pypfopt.equity_curves.to_csv(
        output_dir / f"equity_curve_pypfopt_{cfg.experiment}.csv"
    )

    # Log
    log_data = {
        "step_id": 6,
        "experiment": cfg.experiment,
        "time_window": "out_of_sample",
        "rebal_freq": cfg.backtest.rebal_freq,
        "window_in_sample_months": cfg.backtest.window_in_sample_months,
        "transaction_cost_bps": cfg.backtest.transaction_cost_bps,
        "slippage_bps": cfg.backtest.slippage_bps,
        "metrics": {
            "manual": result_manual.metrics,
            "pypfopt": result_pypfopt.metrics,
        },
    }

    log_path = Path(cfg.paths.reports.logs) / f"backtest_results_{cfg.experiment}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)

    # Also save as CSV
    metrics_df = pd.DataFrame({
        "manual": result_manual.metrics.get("tangency", {}),
        "pypfopt": result_pypfopt.metrics.get("tangency", {}),
    }).T
    metrics_df.to_csv(output_dir / f"backtest_metrics_{cfg.experiment}.csv")

    logger.info(f"Backtest completed. Logged to {log_path}")


if __name__ == "__main__":
    main()

