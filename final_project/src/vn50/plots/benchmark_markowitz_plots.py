"""Visualization utilities for Markowitz benchmark comparison."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_benchmark_table(csv_path: Path) -> pd.DataFrame:
    """Load benchmark table CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark table not found: {csv_path}")

    df = pd.read_csv(csv_path, index_col=0)
    numeric_cols = [
        "ann_return_pct",
        "ann_vol_pct",
        "sharpe",
        "max_drawdown_pct",
        "calmar",
        "sortino",
        "turnover",
        "excess_ann_return_pct",
        "excess_sharpe",
        "vol_ratio",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.index = [idx.replace("(BENCHMARK)", "").strip().title() for idx in df.index]
    df.index = [idx.replace("Equal-Weight", "Equal-weight") for idx in df.index]
    logger.info("Loaded benchmark table with strategies: %s", ", ".join(df.index))
    return df


def plot_risk_return_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot risk-return scatter with Sharpe annotations."""
    required_cols = {"ann_return_pct", "ann_vol_pct", "sharpe"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Benchmark table missing columns: {required_cols - set(df.columns)}")

    fig, ax = plt.subplots(figsize=(10, 7))
    for idx, row in df.iterrows():
        marker = "o"
        size = 120
        color = "#4C72B0"
        if "equal" in idx.lower():
            marker = "s"
            size = 160
            color = "#DD8452"
        ax.scatter(row["ann_vol_pct"], row["ann_return_pct"], s=size, marker=marker, color=color, label=idx)
        ax.annotate(
            f"Sharpe={row['sharpe']:.2f}",
            (row["ann_vol_pct"], row["ann_return_pct"]),
            textcoords="offset points",
            xytext=(5, 8),
        )

    ax.set_title("Risk–Return Benchmark – Markowitz vs Equal-weight (VN30)")
    ax.set_xlabel("Volatility năm hóa (%)")
    ax.set_ylabel("Lợi suất năm hóa (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "benchmark_risk_return_scatter.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved risk-return scatter to %s", output_path)


def plot_excess_return_and_sharpe(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot excess annual return and Sharpe vs benchmark."""
    df_subset = df.dropna(subset=["excess_ann_return_pct", "excess_sharpe"], how="all")
    df_subset = df_subset[[not ("equal" in idx.lower()) for idx in df_subset.index]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Excess Return & Sharpe – Markowitz vs Equal-weight")

    axes[0].bar(df_subset.index, df_subset["excess_ann_return_pct"], color="#4C72B0")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Chênh lệch lợi suất năm hóa so với Equal-weight")
    axes[0].set_ylabel("Điểm %")
    axes[0].grid(True, axis="y", alpha=0.3, linestyle="--")
    for idx, val in enumerate(df_subset["excess_ann_return_pct"]):
        axes[0].text(idx, val, f"{val:.2f}%", ha="center", va="bottom" if val >= 0 else "top")

    axes[1].bar(df_subset.index, df_subset["excess_sharpe"], color="#55A868")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Chênh lệch Sharpe so với Equal-weight")
    axes[1].set_ylabel("Δ Sharpe")
    axes[1].grid(True, axis="y", alpha=0.3, linestyle="--")
    for idx, val in enumerate(df_subset["excess_sharpe"]):
        axes[1].text(idx, val, f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = out_dir / "benchmark_excess_return_sharpe.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved excess return & Sharpe chart to %s", output_path)


def plot_volatility_ratio(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot volatility ratio vs benchmark."""
    if "vol_ratio" not in df.columns:
        raise ValueError("Benchmark table missing 'vol_ratio' column.")

    df_subset = df[[not ("equal" in idx.lower()) for idx in df.index]]
    ratios = df_subset["vol_ratio"].dropna()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(ratios.index, ratios.values, color="#C44E52")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Tỷ lệ biến động so với Equal-weight (vol_ratio)")
    ax.set_ylabel("vol_ratio")
    ax.grid(True, axis="y", alpha=0.3)

    for idx, val in enumerate(ratios.values):
        ax.text(idx, val, f"{val:.2f}", ha="center", va="bottom" if val >= 1 else "top")

    fig.tight_layout()
    output_path = out_dir / "benchmark_vol_ratio.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved volatility ratio chart to %s", output_path)


def save_pretty_benchmark_table(df: pd.DataFrame, out_dir: Path) -> None:
    """Save a clean subset of benchmark metrics (CSV + Markdown)."""
    columns = [
        "ann_return_pct",
        "ann_vol_pct",
        "sharpe",
        "max_drawdown_pct",
        "calmar",
        "sortino",
        "turnover",
    ]
    subset = df.loc[:, [col for col in columns if col in df.columns]]
    csv_path = out_dir / "benchmark_markowitz_table_pretty.csv"
    subset.to_csv(csv_path, index_label="strategy")
    logger.info("Saved pretty benchmark table to %s", csv_path)

    md_path = out_dir / "benchmark_markowitz_table_pretty.md"
    try:
        md_text = subset.to_markdown()
    except ImportError:
        md_text = subset.to_string()
        logger.warning("tabulate not installed; wrote plain text table to %s", md_path)
    md_path.write_text(md_text, encoding="utf-8")

