"""Run the entire VN30 Markowitz pipeline using an existing raw price CSV.

The script treats `data/raw/prices.csv` as the canonical source, mirrors it to
`data/interim/prices_raw.csv`, `data/raw/csv/prices.csv`, and
`data/raw/parquet/prices.parquet`, and then executes:
  1. Cleanup of derived outputs (interim/processed/artifacts/logs)
  2. Data preprocessing
  3. Return generation
  4. EDA reports
  5. Markowitz walk-forward backtest
  6. Benchmark analysis
  7. Efficient frontier plot

Outputs are written under `data/interim`, `data/processed`, and `reports/*`.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.vn50.cli import (
    clean_outputs,
    make_returns,
    plot_efficient_frontier,
    preprocess_data,
    run_benchmark_analysis,
    run_eda,
    run_markowitz,
)

logger = logging.getLogger(__name__)


def _copy_if_different(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    logger.info("Copied %s -> %s", source, destination)


def _prepare_raw_derivatives(cfg: DictConfig) -> None:
    """Ensure downstream steps see the expected raw/interim files."""
    raw_dir = Path(cfg.paths.data.raw)
    canonical_csv = raw_dir / "prices.csv"
    if not canonical_csv.exists():
        raise FileNotFoundError(f"Canonical raw file missing: {canonical_csv}")

    logger.info("Preparing raw derivatives from %s", canonical_csv)

    # Mirror to interim CSV expected by preprocess CLI
    interim_csv = Path(cfg.paths.data.interim) / "prices_raw.csv"
    _copy_if_different(canonical_csv, interim_csv)

    # Mirror to legacy raw/csv location
    raw_csv_dir = raw_dir / "csv"
    _copy_if_different(canonical_csv, raw_csv_dir / "prices.csv")

    # Build parquet version for consumers that prefer parquet
    df = pd.read_csv(canonical_csv, index_col=0, parse_dates=True)
    parquet_dir = raw_dir / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / "prices.parquet"
    df.to_parquet(parquet_path)
    logger.info("Wrote parquet mirror to %s (rows=%d, cols=%d)", parquet_path, *df.shape)


def _run_step(name: str, fn: Callable[[], None]) -> None:
    logger.info("========== %s : START ==========", name)
    fn()
    logger.info("========== %s : DONE ==========", name)


def _hydra_free_call(step_module_main: Callable[[DictConfig], None], cfg: DictConfig) -> None:
    """Invoke a Hydra-decorated main function without re-initializing Hydra."""
    target = getattr(step_module_main, "__wrapped__", step_module_main)
    target(cfg)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main_app(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting full VN30 Markowitz pipeline (experiment=%s)", cfg.experiment)

    _run_step("Cleanup Outputs", clean_outputs.main_app)
    _prepare_raw_derivatives(cfg)

    _run_step(
        "Preprocess Data",
        lambda: _hydra_free_call(preprocess_data.main, cfg),
    )
    _run_step(
        "Make Returns",
        lambda: _hydra_free_call(make_returns.main, cfg),
    )
    _run_step(
        "EDA",
        lambda: _hydra_free_call(run_eda.main, cfg),
    )
    _run_step(
        "Markowitz Walk-Forward",
        lambda: _hydra_free_call(run_markowitz.main, cfg),
    )
    _run_step(
        "Benchmark Analysis",
        lambda: run_benchmark_analysis.main_app(cfg),
    )
    _run_step(
        "Efficient Frontier Plot",
        lambda: plot_efficient_frontier.main_app(cfg),
    )

    logger.info("Full VN30 Markowitz pipeline completed successfully.")


if __name__ == "__main__":
    main_app()

