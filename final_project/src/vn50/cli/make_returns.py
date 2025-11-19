"""CLI: Calculate returns."""

import json
import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.vn50.features import returns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Calculate returns."""
    logger.info(f"Step 3: Calculating returns for experiment: {cfg.experiment}")

    # Load cleaned prices
    input_path = Path(cfg.paths.data.processed) / "prices_clean.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    prices = pd.read_csv(input_path, index_col=0, parse_dates=True)

    # Daily log returns (for EDA)
    returns_daily = returns.make_returns(
        prices,
        kind=cfg.features.returns.kind,
        window=cfg.features.returns.window,
        freq="D",
    )

    # Monthly simple returns (for Markowitz)
    returns_monthly = returns.make_returns(
        prices,
        kind="arith",
        window=1,
        freq="M",
    )

    # Save daily returns (CSV and Parquet)
    output_daily_csv = Path(cfg.paths.data.processed) / "returns_daily_log.csv"
    output_daily_parquet = Path(cfg.paths.data.processed) / "returns_daily_log.parquet"
    output_daily_csv.parent.mkdir(parents=True, exist_ok=True)
    returns_daily.to_csv(output_daily_csv)
    returns_daily.to_parquet(
        output_daily_parquet, compression="snappy", index=True
    )

    # Save monthly returns (CSV and Parquet)
    output_monthly_csv = Path(cfg.paths.data.processed) / "returns_monthly_simple.csv"
    output_monthly_parquet = Path(cfg.paths.data.processed) / "returns_monthly_simple.parquet"
    returns_monthly.to_csv(output_monthly_csv)
    returns_monthly.to_parquet(
        output_monthly_parquet, compression="snappy", index=True
    )

    # Log
    log_data = {
        "step_id": 3,
        "experiment": cfg.experiment,
        "daily_returns": {
            "n_periods": len(returns_daily),
            "n_assets": len(returns_daily.columns),
        },
        "monthly_returns": {
            "n_periods": len(returns_monthly),
            "n_assets": len(returns_monthly.columns),
        },
    }

    log_path = Path(cfg.paths.reports.logs) / f"make_returns_{cfg.experiment}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)

    logger.info(
        f"Saved daily log returns to {output_daily_csv} and {output_daily_parquet}"
    )
    logger.info(
        f"Saved monthly simple returns to {output_monthly_csv} and {output_monthly_parquet}"
    )
    logger.info(f"Logged to {log_path}")


if __name__ == "__main__":
    main()

