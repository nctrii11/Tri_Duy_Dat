"""CLI: Benchmark analysis for Markowitz walk-forward results."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.vn50.analysis import benchmark_markowitz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main_app(cfg: DictConfig) -> None:
    """Generate benchmark comparison outputs."""
    metrics_path = Path(cfg.paths.reports.logs) / "metrics_walkforward_markowitz.json"
    artifacts_dir = Path(cfg.paths.reports.artifacts)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics = benchmark_markowitz.load_markowitz_metrics(metrics_path)
    rel_metrics = benchmark_markowitz.compute_relative_to_benchmark(metrics, benchmark_key="equal_weight")
    table = benchmark_markowitz.build_benchmark_table(metrics, rel_metrics)
    markdown_str = benchmark_markowitz.render_benchmark_markdown(table, rel_metrics)

    table_csv = artifacts_dir / "benchmark_markowitz_table.csv"
    table.to_csv(table_csv)
    summary_md = artifacts_dir / "benchmark_markowitz_summary.md"
    summary_md.write_text(markdown_str, encoding="utf-8")

    logger.info("Saved benchmark table to %s", table_csv)
    logger.info("Saved benchmark summary to %s", summary_md)

    eq_sharpe = metrics.get("equal_weight", {}).get("sharpe")
    logger.info(
        "Sharpe ratios – GMV: %.3f, Tangency: %.3f, Equal-weight: %.3f",
        metrics.get("gmv", {}).get("sharpe", float("nan")),
        metrics.get("tangency", {}).get("sharpe", float("nan")),
        eq_sharpe if eq_sharpe is not None else float("nan"),
    )
    logger.info(
        "Excess annual return vs EW – GMV: %.2f%%, Tangency: %.2f%%",
        rel_metrics.get("gmv", {}).get("excess_ann_return", 0.0) * 100,
        rel_metrics.get("tangency", {}).get("excess_ann_return", 0.0) * 100,
    )


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint."""
    main_app(cfg)


if __name__ == "__main__":
    main()

