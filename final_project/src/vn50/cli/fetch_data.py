"""Data được cào lại từ vnstock cho VN30 giai đoạn 2020-10-30 đến 2025-10-30 để phục vụ cho Markowitz backtest."""

import json
import logging
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from src.vn50.data import fetch, validation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_universe_from_file(universe_path: Path) -> list[str]:
    if not universe_path.exists():
        raise FileNotFoundError(f"Universe file not found: {universe_path}")
    cfg = OmegaConf.load(universe_path)
    data: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    tickers = data.get("tickers")
    if not isinstance(tickers, list) or not all(isinstance(ticker, str) for ticker in tickers):
        raise ValueError(f"Universe file {universe_path} must define a list[str] under 'tickers'")
    return [ticker.upper() for ticker in tickers]


def _resolve_universe(cfg: DictConfig) -> list[str]:
    if cfg.data.get("universe"):
        universe_path = Path(cfg.data.universe)
        tickers = _load_universe_from_file(universe_path)
        logger.info("Loaded %s tickers from %s", len(tickers), universe_path)
        return tickers
    return [ticker.upper() for ticker in cfg.data.tickers]


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Fetch VN30 prices from vnstock and persist them to data/raw for downstream steps."""
    logger.info("Step 1: Fetching data for experiment: %s", cfg.experiment)
    tickers = _resolve_universe(cfg)
    start = cfg.data.get("start", cfg.data.get("start_date"))
    end = cfg.data.get("end", cfg.data.get("end_date"))
    if start is None or end is None:
        raise ValueError("Both cfg.data.start and cfg.data.end (or legacy *_date) must be provided")

    prices = fetch.fetch_prices(
        symbols=tickers,
        start=start,
        end=end,
        adjusted=cfg.data.get("adjusted", True),
        source=cfg.data.get("source", "vnstock"),
        max_retries=cfg.data.get("max_retries", 3),
        sleep_seconds=cfg.data.get("sleep_seconds", 0.5),
    )

    validation.validate_raw_prices(
        prices,
        expected_tickers=tickers,
        start_date=start,
        end_date=end,
    )

    raw_base = Path(cfg.paths.data.raw)
    parquet_path = raw_base / "parquet" / "prices.parquet"
    csv_path = raw_base / "csv" / "prices.csv"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    prices.to_parquet(parquet_path)
    prices.to_csv(csv_path)

    interim_path = Path(cfg.paths.data.interim) / "prices_raw.csv"
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(interim_path)

    log_data = {
        "step_id": 1,
        "experiment": cfg.experiment,
        "source": cfg.data.get("source", "vnstock"),
        "adjusted": cfg.data.get("adjusted", True),
        "n_assets": len(prices.columns),
        "n_days": len(prices),
        "date_range": [str(prices.index[0]), str(prices.index[-1])],
        "tickers": tickers,
    }
    log_path = Path(cfg.paths.reports.logs) / f"fetch_data_{cfg.experiment}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, default=str)

    logger.info("Saved parquet prices to %s", parquet_path)
    logger.info("Saved csv prices to %s", csv_path)
    logger.info("Saved interim prices to %s", interim_path)
    logger.info("Logged fetch summary to %s", log_path)


if __name__ == "__main__":
    main()

