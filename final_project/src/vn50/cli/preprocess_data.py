"""CLI: Preprocess price data."""

import json
import logging
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.vn50.data import preprocess
from src.vn50.features import returns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Preprocess price data."""
    logger.info(f"Step 2: Preprocessing data for experiment: {cfg.experiment}")

    # Load raw prices
    input_path = Path(cfg.paths.data.interim) / "prices_raw.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    prices = pd.read_csv(input_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded raw prices: {len(prices)} days × {len(prices.columns)} tickers")

    # Clean prices
    calendar_path = None
    if cfg.preprocess.align_to_calendar:
        calendar_path = Path(cfg.preprocess.get("calendar_path", "data/cache/hose_trading_days.csv"))

    prices_clean = preprocess.clean_prices(
        prices,
        max_missing_ratio=cfg.data.max_missing_ratio,
        deduplicate=cfg.preprocess.deduplicate,
        align_to_calendar=cfg.preprocess.align_to_calendar,
        calendar_path=calendar_path,
        drop_nonpositive=cfg.preprocess.drop_nonpositive,
        remove_all_nan_days=cfg.preprocess.remove_all_nan_days,
    )

    # Calculate returns
    returns_daily = returns.make_returns(
        prices_clean,
        kind=cfg.features.returns.kind,
        window=cfg.features.returns.window,
        freq="D",
    )

    # Calculate metrics before winsorization
    avg_abs_ret_before = returns_daily.abs().mean().mean()
    pct_days_all_nan = (returns_daily.isna().all(axis=1).sum() / len(returns_daily)) * 100
    avg_missing_ratio = returns_daily.isna().mean().mean() * 100

    # Winsorize returns (not prices)
    if cfg.preprocess.winsorization.enable:
        returns_daily = preprocess.winsorize_returns(
            returns_daily,
            mode=cfg.preprocess.winsorization.mode,
            window=cfg.preprocess.winsorization.window,
            lower_q=cfg.preprocess.winsorization.lower_q,
            upper_q=cfg.preprocess.winsorization.upper_q,
            cap_by_exchange_limit=cfg.preprocess.winsorization.get("cap_by_exchange_limit", False),
            abs_cap=cfg.preprocess.winsorization.get("abs_cap"),
        )

    # Calculate metrics after winsorization
    avg_abs_ret_after = returns_daily.abs().mean().mean()
    
    # Calculate % returns clipped
    returns_before_winsor = returns.make_returns(
        prices_clean,
        kind=cfg.features.returns.kind,
        window=cfg.features.returns.window,
        freq="D",
    )
    returns_before_winsor = returns_before_winsor.dropna(how="all")
    pct_returns_clipped = (
        (returns_daily != returns_before_winsor.loc[returns_daily.index]).sum().sum() /
        (len(returns_daily) * len(returns_daily.columns))
    ) * 100

    # Stale masking (on returns)
    if cfg.preprocess.stale_masking.enable:
        returns_before_stale = returns_daily.copy()
        warn_threshold = cfg.preprocess.stale_masking.get("warn_threshold_pct", 10.0)
        returns_daily = preprocess.mask_stale_returns(
            returns_daily,
            min_consecutive_days=cfg.preprocess.stale_masking.min_consecutive_days,
            warn_threshold_pct=warn_threshold,
        )
        pct_returns_masked = (
            returns_daily.isna().sum().sum() - returns_before_stale.isna().sum().sum()
        ) / (len(returns_daily) * len(returns_daily.columns)) * 100
    else:
        pct_returns_masked = 0.0

    # Calculate monthly simple returns for Markowitz
    returns_monthly = returns.make_returns(
        prices_clean,
        kind="arith",
        window=1,
        freq="M",
    )

    # Apply same winsorization and stale masking to monthly returns
    if cfg.preprocess.winsorization.enable:
        returns_monthly = preprocess.winsorize_returns(
            returns_monthly,
            mode=cfg.preprocess.winsorization.mode,
            window=min(12, len(returns_monthly)),  # 12 months for monthly data
            lower_q=cfg.preprocess.winsorization.lower_q,
            upper_q=cfg.preprocess.winsorization.upper_q,
            cap_by_exchange_limit=cfg.preprocess.winsorization.get("cap_by_exchange_limit", False),
            abs_cap=cfg.preprocess.winsorization.get("abs_cap"),
        )

    if cfg.preprocess.stale_masking.enable:
        returns_monthly = preprocess.mask_stale_returns(
            returns_monthly,
            min_consecutive_days=max(1, cfg.preprocess.stale_masking.min_consecutive_days // 20),  # Scale for monthly
            warn_threshold_pct=warn_threshold,
        )

    # Save cleaned prices (CSV and Parquet)
    output_prices_csv = Path(cfg.paths.data.processed) / "prices_clean.csv"
    output_prices_parquet = Path(cfg.paths.data.processed) / "prices_clean.parquet"
    output_prices_csv.parent.mkdir(parents=True, exist_ok=True)
    prices_clean.to_csv(output_prices_csv)
    prices_clean.to_parquet(
        output_prices_parquet, compression="snappy", index=True
    )

    # Save cleaned daily returns (CSV and Parquet)
    output_returns_daily_csv = Path(cfg.paths.data.processed) / "returns_daily_log.csv"
    output_returns_daily_parquet = Path(cfg.paths.data.processed) / "returns_daily_log.parquet"
    returns_daily.to_csv(output_returns_daily_csv)
    returns_daily.to_parquet(
        output_returns_daily_parquet, compression="snappy", index=True
    )

    # Save cleaned monthly returns for Markowitz (CSV and Parquet)
    output_returns_monthly_csv = Path(cfg.paths.data.processed) / "returns_monthly_simple.csv"
    output_returns_monthly_parquet = Path(cfg.paths.data.processed) / "returns_monthly_simple.parquet"
    returns_monthly.to_csv(output_returns_monthly_csv)
    returns_monthly.to_parquet(
        output_returns_monthly_parquet, compression="snappy", index=True
    )

    # Prepare log data
    log_data = {
        "step_id": 2,
        "experiment": cfg.experiment,
        "timestamp": datetime.now().isoformat(),
        "n_assets": len(prices_clean.columns),
        "n_days": len(prices_clean),
        "date_range": [str(prices_clean.index[0]), str(prices_clean.index[-1])],
        "tickers": list(prices_clean.columns),
        "params": {
            "max_missing_ratio": float(cfg.data.max_missing_ratio),
            "deduplicate": cfg.preprocess.deduplicate,
            "align_to_calendar": cfg.preprocess.align_to_calendar,
            "drop_nonpositive": cfg.preprocess.drop_nonpositive,
            "remove_all_nan_days": cfg.preprocess.remove_all_nan_days,
            "winsor": {
                "enable": cfg.preprocess.winsorization.enable,
                "mode": cfg.preprocess.winsorization.mode,
                "window": cfg.preprocess.winsorization.window,
                "lower_q": cfg.preprocess.winsorization.lower_q,
                "upper_q": cfg.preprocess.winsorization.upper_q,
                "cap_by_exchange_limit": cfg.preprocess.winsorization.get("cap_by_exchange_limit", False),
                "abs_cap": cfg.preprocess.winsorization.get("abs_cap"),
            },
            "stale": {
                "enable": cfg.preprocess.stale_masking.enable,
                "min_consecutive_days": cfg.preprocess.stale_masking.min_consecutive_days,
            },
            "returns_kind": cfg.features.returns.kind,
            "returns_window": cfg.features.returns.window,
        },
        "metrics": {
            "pct_days_all_nan": float(pct_days_all_nan),
            "avg_missing_ratio": float(avg_missing_ratio),
            "avg_abs_ret_before": float(avg_abs_ret_before),
            "avg_abs_ret_after": float(avg_abs_ret_after),
            "pct_returns_masked": float(pct_returns_masked),
            "pct_returns_clipped": float(pct_returns_clipped),
        },
    }

    log_path = Path(cfg.paths.reports.logs) / f"preprocess_{cfg.experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)

    logger.info(f"Saved cleaned prices to {output_prices_csv} and {output_prices_parquet}")
    logger.info(f"Saved daily log returns to {output_returns_daily_csv} and {output_returns_daily_parquet}")
    logger.info(f"Saved monthly simple returns to {output_returns_monthly_csv} and {output_returns_monthly_parquet}")
    logger.info(f"Logged to {log_path}")
    logger.info(f"Metrics: avg_missing_ratio={avg_missing_ratio:.2f}%, "
                f"avg_abs_ret_before={avg_abs_ret_before:.4f}, "
                f"avg_abs_ret_after={avg_abs_ret_after:.4f}, "
                f"pct_returns_clipped={pct_returns_clipped:.2f}%, "
                f"pct_returns_masked={pct_returns_masked:.2f}%")


if __name__ == "__main__":
    main()
