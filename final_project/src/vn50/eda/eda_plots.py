"""Minimal EDA: generate the five mandated VN30 plots only."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("reports/eda")
FIGSIZE = (40, 16)
DPI = 300
TITLE_FONTSIZE = 30
LABEL_FONTSIZE = 22
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 18


def _load_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def _prepare_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _plot_price_history_actual(prices: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ticker in prices.columns:
        ax.plot(prices.index, prices[ticker], linewidth=1.2, alpha=0.9, label=ticker)
    ax.set_title("VN30 Price History – Nominal Closing Prices (VND)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (VND)")
    ax.legend(ncol=3, fontsize=10, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "price_history_actual.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    normalized = prices.copy()
    for col in normalized.columns:
        series = normalized[col].dropna()
        if series.empty:
            normalized[col] = np.nan
        else:
            normalized[col] = normalized[col] / series.iloc[0] * 100
    return normalized


def _plot_price_history_normalized(prices: pd.DataFrame) -> None:
    normalized = _normalize_prices(prices)
    vn30_avg = normalized.mean(axis=1)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ticker in normalized.columns:
        ax.plot(normalized.index, normalized[ticker], linewidth=1.0, alpha=0.6)
    ax.plot(vn30_avg.index, vn30_avg, color="black", linewidth=3, label="VN30 Average")
    ax.set_title("VN30 Price History – Normalized to 100")
    ax.set_xlabel("Date")
    ax.set_ylabel("Indexed Price (Base=100)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "price_history_normalized.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_rolling_volatility(rets_daily: pd.DataFrame, window: int = 63) -> None:
    rolling_std = rets_daily.rolling(window=window, min_periods=window).std()
    annualized = rolling_std * np.sqrt(252)
    average_vol = annualized.mean(axis=1)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ticker in annualized.columns:
        ax.plot(annualized.index, annualized[ticker], color="lightsteelblue", linewidth=0.9, alpha=0.7)
    ax.plot(average_vol.index, average_vol, color="black", linewidth=3, label="VN30 Average")
    ax.set_title(f"{window}-Day Rolling Volatility (Annualized)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rolling_volatility.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_monthly_return_histogram(rets_monthly: pd.DataFrame) -> None:
    combined = rets_monthly.values.ravel()
    combined = combined[np.isfinite(combined)]
    mean_val = combined.mean()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(combined, bins=80, color="lightcoral", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero")
    ax.axvline(mean_val, color="blue", linestyle="--", linewidth=2, label=f"Mean ({mean_val:.4f})")
    ax.set_title("Monthly Return Distribution – All VN30 Constituents")
    ax.set_xlabel("Monthly Simple Return")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "monthly_returns_hist.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_covariance_eigen_analysis(rets_monthly: pd.DataFrame) -> None:
    cleaned = rets_monthly.dropna()
    cov = cleaned.cov()
    eigvals = np.linalg.eigvalsh(cov.values)
    eigvals = np.sort(eigvals)[::-1]
    explained = eigvals / eigvals.sum()
    cumulative = np.cumsum(explained)
    indices = np.arange(1, len(eigvals) + 1)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    axes[0].plot(indices, eigvals, marker="o", linewidth=2, color="steelblue")
    axes[0].set_title("Covariance Eigenvalues – Scree Plot")
    axes[0].set_xlabel("Eigenvalue Index")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(indices, explained * 100, marker="o", linewidth=2, color="seagreen", label="Individual")
    axes[1].plot(indices, cumulative * 100, marker="s", linewidth=2, color="darkorange", label="Cumulative")
    axes[1].axhline(80, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_title("Explained Variance by Eigenvalues")
    axes[1].set_xlabel("Eigenvalue Index")
    axes[1].set_ylabel("Variance Explained (%)")
    axes[1].legend(frameon=False)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "covariance_eigen_analysis.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def run_all_eda() -> None:
    """Generate the five required EDA plots from processed VN30 datasets."""
    logging.basicConfig(level=logging.INFO)
    sns.set_theme(style="whitegrid")
    _prepare_output_dir()

    prices = _load_csv("prices_clean.csv")
    daily_returns = _load_csv("returns_daily_log.csv")
    monthly_returns = _load_csv("returns_monthly_simple.csv")

    _plot_price_history_actual(prices)
    _plot_price_history_normalized(prices)
    _plot_rolling_volatility(daily_returns)
    _plot_monthly_return_histogram(monthly_returns)
    _plot_covariance_eigen_analysis(monthly_returns)

    logger.info("EDA plots saved to %s", OUTPUT_DIR.resolve())


def _plot_monthly_distribution_combo(rets_monthly: pd.DataFrame) -> None:
    combined = rets_monthly.values.ravel()
    combined = combined[np.isfinite(combined)]
    mean_val = combined.mean()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    axes[0].hist(combined, bins=80, color="forestgreen", edgecolor="white", alpha=0.85, density=True)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=2, label="Zero return")
    axes[0].axvline(mean_val, color="blue", linestyle="--", linewidth=2, label=f"Mean ({mean_val:.4f})")
    axes[0].set_title("Monthly Simple Returns Distribution – All VN30", fontsize=TITLE_FONTSIZE)
    axes[0].set_xlabel("Simple Return", fontsize=LABEL_FONTSIZE)
    axes[0].set_ylabel("Density", fontsize=LABEL_FONTSIZE)
    axes[0].tick_params(axis="both", labelsize=TICK_FONTSIZE)
    axes[0].legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    axes[0].grid(True, alpha=0.2)

    subset = rets_monthly.iloc[:, :15]
    bp = axes[1].boxplot(
        [subset[col].dropna().values for col in subset.columns],
        labels=subset.columns,
        vert=True,
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_title("Monthly Returns by Ticker (First 15)", fontsize=TITLE_FONTSIZE)
    axes[1].set_xlabel("Ticker", fontsize=LABEL_FONTSIZE)
    axes[1].set_ylabel("Simple Return", fontsize=LABEL_FONTSIZE)
    axes[1].tick_params(axis="x", rotation=45, labelsize=TICK_FONTSIZE)
    axes[1].tick_params(axis="y", labelsize=TICK_FONTSIZE)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Monthly Simple Returns Analysis (Markowitz Input)", fontsize=TITLE_FONTSIZE + 6, y=0.97)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "monthly_returns_distribution.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_prices_actual_named(prices: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ticker in prices.columns:
        ax.plot(prices.index, prices[ticker], linewidth=1.0, label=ticker)
    ax.set_title("Actual Closing Prices – All 30 VN30 Tickers (Nominal VND)", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Closing Price (VND)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(ncol=3, fontsize=LEGEND_FONTSIZE, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "prices_actual_vn30.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_corr_heatmap_monthly(rets_monthly: pd.DataFrame) -> None:
    corr = rets_monthly.corr()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.heatmap(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )
    ax.set_title("Correlation Matrix – VN30 Monthly Returns (Markowitz Input)", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Ticker", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Ticker", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    if ax.figure.axes:
        colorbar = ax.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=TICK_FONTSIZE)
        colorbar.set_label("Correlation", fontsize=LABEL_FONTSIZE)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "corr_heatmap_vn30_monthly.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def run_requested_charts() -> None:
    """Rebuild the monthly distribution, price history, and correlation heatmap figures."""
    logging.basicConfig(level=logging.INFO)
    sns.set_theme(style="whitegrid")
    _prepare_output_dir()

    prices = _load_csv("prices_clean.csv")
    monthly_returns = _load_csv("returns_monthly_simple.csv")

    _plot_monthly_distribution_combo(monthly_returns)
    _plot_prices_actual_named(prices)
    _plot_corr_heatmap_monthly(monthly_returns)

    logger.info("Requested charts saved to %s", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    run_all_eda()

