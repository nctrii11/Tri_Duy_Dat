"""CLI: Plot Markowitz efficient frontier for VN30."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.vn50.plots import efficient_frontier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main_app(cfg: DictConfig) -> None:
    """Compute mu/Sigma and plot efficient frontier."""
    mu, Sigma = efficient_frontier.compute_mu_sigma_from_cfg(cfg)
    artifacts_dir = Path(cfg.paths.reports.artifacts)
    output_path = efficient_frontier.plot_efficient_frontier_static(mu, Sigma, cfg, artifacts_dir)

    logger.info(
        "Efficient frontier plotted for %d assets. Output: %s",
        len(mu),
        output_path,
    )


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    main_app(cfg)


if __name__ == "__main__":
    main()


