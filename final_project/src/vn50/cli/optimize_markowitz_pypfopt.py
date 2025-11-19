"""CLI: Optimize Markowitz portfolio (PyPortfolioOpt)."""

import json
import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.vn50.split import time
from src.vn50.optimize import markowitz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Optimize Markowitz portfolio (PyPortfolioOpt)."""
    logger.info(f"Step 5: Markowitz optimization (PyPortfolioOpt) for experiment: {cfg.experiment}")

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

    # Optimize
    weights_dict = markowitz.optimize_markowitz_pypfopt(
        in_sample_returns,
        risk_model=cfg.markowitz_pypfopt.risk_model,
        objective=cfg.markowitz_pypfopt.objective,
        risk_free_rate=cfg.markowitz_manual.risk_free_rate,  # Use same RF rate
        weight_bounds=(
            cfg.markowitz_pypfopt.weight_bounds.min,
            cfg.markowitz_pypfopt.weight_bounds.max,
        ),
        L2_penalty=cfg.markowitz_pypfopt.L2_penalty,
    )

    # Save weights
    output_dir = Path(cfg.paths.reports.artifacts)
    output_dir.mkdir(parents=True, exist_ok=True)

    for strategy_name, weights in weights_dict.items():
        if isinstance(weights, pd.Series):
            output_path = output_dir / f"weights_pypfopt_{strategy_name}_{cfg.experiment}.csv"
            weights.to_csv(output_path)
            logger.info(f"Saved {strategy_name} weights to {output_path}")

    # Calculate metrics (simplified, PyPO has its own metrics)
    metrics_dict = {}
    for strategy_name, weights in weights_dict.items():
        if isinstance(weights, pd.Series):
            # Simple metrics calculation
            portfolio_return = (in_sample_returns.mean() * 12 * weights).sum()
            portfolio_vol = (weights @ in_sample_returns.cov() * 12 @ weights) ** 0.5
            sharpe = (portfolio_return - cfg.markowitz_manual.risk_free_rate) / portfolio_vol

            metrics_dict[strategy_name] = {
                "ann_return": float(portfolio_return),
                "ann_vol": float(portfolio_vol),
                "sharpe": float(sharpe),
            }

    # Log
    log_data = {
        "step_id": 5,
        "experiment": cfg.experiment,
        "time_window": "in_sample",
        "n_assets": len(in_sample_returns.columns),
        "n_periods": len(in_sample_returns),
        "risk_model": cfg.markowitz_pypfopt.risk_model,
        "objective": cfg.markowitz_pypfopt.objective,
        "risk_free_rate": cfg.markowitz_manual.risk_free_rate,
        "weight_bounds": [cfg.markowitz_pypfopt.weight_bounds.min, cfg.markowitz_pypfopt.weight_bounds.max],
        "L2_penalty": cfg.markowitz_pypfopt.L2_penalty,
        "metrics": metrics_dict,
    }

    log_path = Path(cfg.paths.reports.logs) / f"markowitz_pypfopt_{cfg.experiment}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)

    logger.info(f"Logged to {log_path}")


if __name__ == "__main__":
    main()

