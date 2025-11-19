"""CLI: Visualize benchmark comparisons for Markowitz strategies."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.vn50.plots import benchmark_markowitz_plots as plots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main_app(cfg: DictConfig) -> None:
    artifacts_dir = Path(cfg.paths.reports.artifacts)
    csv_path = artifacts_dir / "benchmark_markowitz_table.csv"

    df = plots.load_benchmark_table(csv_path)

    plots.plot_risk_return_scatter(df, artifacts_dir)
    plots.plot_excess_return_and_sharpe(df, artifacts_dir)
    plots.plot_volatility_ratio(df, artifacts_dir)
    plots.save_pretty_benchmark_table(df, artifacts_dir)

    logger.info("Benchmark visualization assets written to %s", artifacts_dir.resolve())


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    main_app(cfg)


if __name__ == "__main__":
    main()


