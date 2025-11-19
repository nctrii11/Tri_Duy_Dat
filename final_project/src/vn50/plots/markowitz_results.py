"""Visualization utilities for Markowitz walk-forward backtest results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def load_backtest_results(cfg: DictConfig) -> tuple[pd.DataFrame, dict]:
    """Load walk-forward monthly returns and metrics."""
    artifacts_dir = Path(cfg.paths.reports.artifacts)
    logs_dir = Path(cfg.paths.reports.logs)

    returns_path = artifacts_dir / "returns_walkforward_markowitz.csv"
    metrics_path = logs_dir / "metrics_walkforward_markowitz.json"

    if not returns_path.exists():
        raise FileNotFoundError(f"Returns CSV not found: {returns_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics JSON not found: {metrics_path}")

    df_returns = pd.read_csv(returns_path, index_col=0)
    df_returns.index = pd.to_datetime(df_returns.index)
    df_returns = df_returns.sort_index()

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_payload = json.load(f)
    metrics = metrics_payload.get("metrics", {})

    logger.info(
        "Loaded %d months of returns and metrics for %d strategies",
        len(df_returns),
        len(metrics),
    )
    return df_returns, metrics


def _compute_nav_from_returns(df_returns: pd.DataFrame) -> pd.DataFrame:
    """Convert monthly returns to NAV (start at 1.0) for each strategy."""
    if df_returns.empty:
        raise ValueError("Return series for NAV computation is empty.")

    nav = (1 + df_returns).cumprod()
    nav = pd.concat(
        [
            pd.DataFrame(
                np.ones((1, nav.shape[1])),
                columns=nav.columns,
                index=[df_returns.index[0] - pd.offsets.MonthEnd()],
            ),
            nav,
        ]
    )
    return nav


def plot_nav_curves(df_returns: pd.DataFrame, out_dir: Path, cfg: DictConfig) -> None:
    """Plot NAV curves for GMV, Tangency, Equal-weight."""
    nav = _compute_nav_from_returns(df_returns)

    plt.figure(figsize=(14, 6))
    for column in nav.columns:
        plt.plot(nav.index, nav[column], label=column.upper(), linewidth=2)

    plt.title("Walk-forward Backtest – Markowitz vs Equal-weight (VN30)")
    plt.xlabel("Date")
    plt.ylabel("NAV (start = 1.0)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "nav_walkforward_markowitz.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved NAV chart to %s", output_path)


def plot_metric_bars(metrics: dict, out_dir: Path, cfg: DictConfig) -> None:
    """Plot bar charts comparing ann. return, ann. vol, Sharpe."""
    required_metrics = ["ann_return", "ann_vol", "sharpe"]
    strategies = list(metrics.keys())

    if not strategies:
        raise ValueError("No strategy metrics provided.")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Walk-forward Metrics – Markowitz vs Equal-weight (VN30)")

    for idx, metric_name in enumerate(required_metrics):
        values = []
        for strat in strategies:
            value = metrics[strat].get(metric_name, np.nan)
            if metric_name in {"ann_return", "ann_vol"}:
                value = value * 100
            values.append(value)

        axes[idx].bar([s.upper() for s in strategies], values, color="#4C72B0")
        axes[idx].set_title(metric_name.replace("_", " ").title())
        axes[idx].grid(True, axis="y", alpha=0.3, linestyle="--")

        for tick, val in zip(axes[idx].get_xticks(), values):
            label = f"{val:.1f}%" if metric_name in {"ann_return", "ann_vol"} else f"{val:.2f}"
            axes[idx].text(tick, val, label, ha="center", va="bottom", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = out_dir / "metrics_walkforward_markowitz.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved metrics bar chart to %s", output_path)


def plot_monthly_return_distribution(
    df_returns: pd.DataFrame,
    out_dir: Path,
    cfg: DictConfig,
) -> None:
    """Plot histogram + boxplot for monthly out-of-sample returns."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Monthly Out-of-sample Returns – Walk-forward Backtest")

    combined = df_returns.values.flatten()
    combined = combined[np.isfinite(combined)]
    mean_val = combined.mean()

    axes[0].hist(combined, bins=40, density=True, color="#55A868", edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero")
    axes[0].axvline(mean_val, color="blue", linestyle="--", linewidth=1.5, label=f"Mean ({mean_val:.3%})")
    axes[0].set_title("Histogram (All Strategies)")
    axes[0].set_xlabel("Monthly Return")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].boxplot(
        [df_returns[col].dropna() for col in df_returns.columns],
        labels=[col.upper() for col in df_returns.columns],
        patch_artist=True,
    )
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_title("Boxplot by Strategy")
    axes[1].set_ylabel("Monthly Return")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = out_dir / "monthly_returns_walkforward_markowitz.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved monthly return distribution chart to %s", output_path)


def save_metrics_table(metrics: dict, out_dir: Path, cfg: DictConfig) -> None:
    """Persist metrics table as CSV (and Markdown)."""
    if not metrics:
        logger.warning("No metrics to save.")
        return

    df_metrics = pd.DataFrame(metrics).T
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "metrics_walkforward_markowitz_table.csv"
    df_metrics.to_csv(csv_path, index_label="strategy")
    logger.info("Saved metrics table to %s", csv_path)

    md_path = out_dir / "metrics_walkforward_markowitz_table.md"
    try:
        df_metrics.to_markdown(md_path)
        logger.info("Saved metrics markdown table to %s", md_path)
    except ImportError:
        md_path.write_text(df_metrics.to_string())
        logger.warning("tabulate not installed; wrote plain text table to %s", md_path)
