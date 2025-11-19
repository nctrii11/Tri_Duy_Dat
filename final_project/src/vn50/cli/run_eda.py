"""CLI: Run Exploratory Data Analysis for VN30 Markowitz."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.vn50.features import eda

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run EDA for VN30 Markowitz portfolio optimization."""
    logger.info(f"Running EDA for experiment: {cfg.experiment}")

    # Create output directories
    figures_path = Path(cfg.paths.reports.figures)
    figures_path.mkdir(parents=True, exist_ok=True)

    eda_path = Path(cfg.paths.reports.artifacts) / "eda"
    eda_path.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading EDA data...")
    prices, rets_daily, rets_monthly = eda.load_eda_data(cfg)

    # 1. Summary statistics
    logger.info("Computing summary statistics for Markowitz...")
    summary = eda.summarize_returns_for_markowitz(rets_monthly)
    summary_path = eda_path / "summary_monthly_returns.csv"
    summary.to_csv(summary_path)
    logger.info(f"Saved summary statistics to {summary_path}")

    # 2. Price series (normalized)
    logger.info("Plotting normalized price series...")
    price_plot_path = figures_path / "prices_history_vn30.png"
    eda.plot_price_series(prices, cfg, output_path=price_plot_path)

    # 2b. Actual prices (not normalized)
    logger.info("Plotting actual closing prices...")
    actual_price_plot_path = figures_path / "prices_actual_vn30.png"
    eda.plot_actual_prices(prices, cfg, output_path=actual_price_plot_path)

    # 3. Daily returns distribution
    logger.info("Plotting daily returns distribution...")
    daily_dist_path = figures_path / "daily_returns_distribution.png"
    eda.plot_distribution_daily_returns(rets_daily, output_path=daily_dist_path)

    # 4. Monthly returns distribution
    logger.info("Plotting monthly returns distribution...")
    monthly_dist_path = figures_path / "monthly_returns_distribution.png"
    eda.plot_distribution_monthly_returns(rets_monthly, output_path=monthly_dist_path)

    # 5. Correlation heatmap
    logger.info("Plotting correlation heatmap...")
    corr_plot_path = figures_path / "corr_heatmap_vn30_monthly.png"
    eda.plot_correlation_heatmap(rets_monthly, output_path=corr_plot_path)

    # 6. Covariance eigenvalues
    logger.info("Plotting covariance eigenvalues...")
    eigenvals_plot_path = figures_path / "cov_eigenvalues_vn30.png"
    eda.plot_cov_eigenvalues(rets_monthly, output_path=eigenvals_plot_path)

    # 7. Rolling volatility
    logger.info("Plotting rolling volatility...")
    vol_plot_path = figures_path / "rolling_volatility_vn30.png"
    eda.plot_rolling_volatility(rets_daily, window=63, output_path=vol_plot_path)

    # Summary
    logger.info("=" * 60)
    logger.info("EDA completed successfully!")
    logger.info("=" * 60)
    logger.info(f"Summary statistics: {summary_path}")
    logger.info(f"Figures saved to: {figures_path}")
    logger.info("Generated figures:")
    logger.info(f"  - {price_plot_path.name}")
    logger.info(f"  - {actual_price_plot_path.name}")
    logger.info(f"  - {daily_dist_path.name}")
    logger.info(f"  - {monthly_dist_path.name}")
    logger.info(f"  - {corr_plot_path.name}")
    logger.info(f"  - {eigenvals_plot_path.name}")
    logger.info(f"  - {vol_plot_path.name}")


if __name__ == "__main__":
    main()

