"""CLI: Optimize Markowitz portfolio (manual implementation)."""

import json
import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.vn50.markowitz import estimators
from src.vn50.optimize import markowitz
from src.vn50.split import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Optimize Markowitz portfolio (manual)."""
    logger.info(f"Step 4: Markowitz optimization (manual) for experiment: {cfg.experiment}")

    # Load monthly returns
    input_path = Path(cfg.paths.data.processed) / "returns_monthly.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    returns = pd.read_csv(input_path, index_col=0, parse_dates=True)

    # Split time
    splits = time.split_time(
        returns,
        in_sample_years=cfg.split.in_sample_years,
        out_of_sample_years=cfg.split.out_of_sample_years,
    )

    in_sample_returns = splits["in_sample"]

    # Estimate mu and Sigma
    mu, Sigma = estimators.estimate_mu_sigma(
        in_sample_returns,
        shrinkage=cfg.features.cov.shrinkage,
        shrinkage_factor=cfg.features.cov.shrinkage_factor,
        jitter=cfg.features.cov.jitter,
        annualization_factor=12,  # Monthly returns
    )

    # Optimize
    weights_dict = markowitz.mean_variance_weights(
        mu,
        Sigma,
        objective=cfg.markowitz_manual.objective,
        risk_free_rate=cfg.markowitz_manual.risk_free_rate,
        min_weight=cfg.markowitz_manual.min_weight,
        max_weight=cfg.markowitz_manual.max_weight,
        solver=cfg.markowitz_manual.solver,
        max_iter=cfg.markowitz_manual.max_iter,
    )

    # Save weights
    output_dir = Path(cfg.paths.reports.artifacts)
    output_dir.mkdir(parents=True, exist_ok=True)

    for strategy_name, weights in weights_dict.items():
        if isinstance(weights, pd.Series):
            output_path = output_dir / f"weights_manual_{strategy_name}_{cfg.experiment}.csv"
            weights.to_csv(output_path)
            logger.info(f"Saved {strategy_name} weights to {output_path}")

    # Calculate metrics for each strategy
    metrics_dict = {}
    for strategy_name, weights in weights_dict.items():
        if isinstance(weights, pd.Series):
            portfolio_return = (mu * weights).sum()
            portfolio_vol = (weights @ Sigma @ weights) ** 0.5
            sharpe = (portfolio_return - cfg.markowitz_manual.risk_free_rate) / portfolio_vol

            metrics_dict[strategy_name] = {
                "ann_return": float(portfolio_return),
                "ann_vol": float(portfolio_vol),
                "sharpe": float(sharpe),
            }

    # Log
    log_data = {
        "step_id": 4,
        "experiment": cfg.experiment,
        "time_window": "in_sample",
        "n_assets": len(mu),
        "n_periods": len(in_sample_returns),
        "risk_free_rate": cfg.markowitz_manual.risk_free_rate,
        "bounds": [cfg.markowitz_manual.min_weight, cfg.markowitz_manual.max_weight],
        "solver": cfg.markowitz_manual.solver,
        "shrinkage": cfg.features.cov.shrinkage,
        "metrics": metrics_dict,
    }

    log_path = Path(cfg.paths.reports.logs) / f"markowitz_manual_{cfg.experiment}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)

    logger.info(f"Logged to {log_path}")


if __name__ == "__main__":
    main()

