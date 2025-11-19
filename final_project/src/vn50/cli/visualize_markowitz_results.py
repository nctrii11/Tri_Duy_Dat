"""CLI: Visualize Markowitz walk-forward results."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.vn50.plots import markowitz_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main_app(cfg: DictConfig) -> None:
    """Load backtest artifacts and generate visualization assets."""
    logger.info("Visualizing Markowitz backtest for experiment=%s", cfg.experiment)

    df_returns, metrics = markowitz_results.load_backtest_results(cfg)
    artifacts_dir = Path(cfg.paths.reports.artifacts)

    markowitz_results.plot_nav_curves(df_returns, artifacts_dir, cfg)
    markowitz_results.plot_metric_bars(metrics, artifacts_dir, cfg)
    markowitz_results.plot_monthly_return_distribution(df_returns, artifacts_dir, cfg)
    markowitz_results.save_metrics_table(metrics, artifacts_dir, cfg)

    logger.info("Visualization artifacts saved to %s", artifacts_dir.resolve())


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint."""
    main_app(cfg)


if __name__ == "__main__":
    main()

