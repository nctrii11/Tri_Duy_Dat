"""Exploratory Data Analysis for VN30 Markowitz Portfolio Optimization."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def load_eda_data(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load EDA data from processed files.

    Args:
        cfg: Hydra config

    Returns:
        Tuple of (prices, rets_daily, rets_monthly)
    """
    processed_path = Path(cfg.paths.data.processed)

    # Load prices
    prices_path = processed_path / "prices_clean.csv"
    if not prices_path.exists():
        raise FileNotFoundError(f"Prices file not found: {prices_path}")
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    prices = prices.sort_index()

    # Load daily log returns
    daily_path = processed_path / "returns_daily_log.csv"
    if not daily_path.exists():
        raise FileNotFoundError(f"Daily returns file not found: {daily_path}")
    rets_daily = pd.read_csv(daily_path, index_col=0, parse_dates=True)
    rets_daily = rets_daily.sort_index()

    # Load monthly simple returns
    monthly_path = processed_path / "returns_monthly_simple.csv"
    if not monthly_path.exists():
        raise FileNotFoundError(f"Monthly returns file not found: {monthly_path}")
    rets_monthly = pd.read_csv(monthly_path, index_col=0, parse_dates=True)
    rets_monthly = rets_monthly.sort_index()

    # Validate columns match universe
    expected_tickers = set(cfg.data.tickers)
    prices_tickers = set(prices.columns)
    if prices_tickers != expected_tickers:
        missing = expected_tickers - prices_tickers
        extra = prices_tickers - expected_tickers
        logger.warning(
            f"Ticker mismatch: missing={missing}, extra={extra}. "
            "Using available tickers."
        )

    logger.info(
        f"Loaded EDA data: prices={len(prices)} days, "
        f"daily_returns={len(rets_daily)} periods, "
        f"monthly_returns={len(rets_monthly)} periods"
    )

    return prices, rets_daily, rets_monthly


def summarize_returns_for_markowitz(
    rets_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create summary statistics for monthly returns (input to Markowitz).

    Args:
        rets_monthly: Monthly simple returns DataFrame

    Returns:
        DataFrame with statistics per ticker
    """
    summary = pd.DataFrame(index=rets_monthly.columns)

    # Basic statistics
    summary["mean_monthly_return"] = rets_monthly.mean()
    summary["std_monthly_return"] = rets_monthly.std()
    summary["min_monthly_return"] = rets_monthly.min()
    summary["max_monthly_return"] = rets_monthly.max()

    # Annualized metrics
    summary["ann_return"] = summary["mean_monthly_return"] * 12
    summary["ann_vol"] = summary["std_monthly_return"] * np.sqrt(12)

    # Higher moments
    summary["skew"] = rets_monthly.skew()
    summary["kurtosis"] = rets_monthly.kurtosis()

    # Additional metrics
    summary["sharpe_ratio"] = (
        summary["ann_return"] / summary["ann_vol"]
        if (summary["ann_vol"] > 0).any()
        else np.nan
    )

    logger.info(f"Computed summary statistics for {len(summary)} tickers")

    return summary


def plot_price_series(
    prices: pd.DataFrame,
    cfg: DictConfig,
    output_path: Path | str | None = None,
    n_tickers: int | None = None,
) -> None:
    """
    Plot price series for all VN30 tickers with colors.

    Args:
        prices: Prices DataFrame
        cfg: Config
        output_path: Output path for figure
        n_tickers: Number of tickers to plot (None = all 30 tickers)
    """
    # Normalize prices to start at 100 for better comparison
    prices_normalized = prices / prices.iloc[0] * 100

    # Compute average
    avg_price = prices_normalized.mean(axis=1)

    fig, ax = plt.subplots(figsize=(16, 8))

    # Use colormap for colors
    cmap = plt.cm.get_cmap("tab20")
    colors = [cmap(i / len(prices.columns)) for i in range(len(prices.columns))]

    # Plot all tickers with colors
    if n_tickers is None or n_tickers >= len(prices.columns):
        # Plot all 30 tickers with different colors
        for idx, ticker in enumerate(prices.columns):
            ax.plot(
                prices_normalized.index,
                prices_normalized[ticker],
                alpha=0.6,
                linewidth=1.2,
                color=colors[idx],
                label=ticker,
            )
        title_suffix = "All 30 VN30 Tickers"
    else:
        # Select representative tickers
        returns = prices.pct_change().dropna()
        vol = returns.std()
        selected_tickers = vol.nlargest(n_tickers).index.tolist()
        for idx, ticker in enumerate(selected_tickers):
            ticker_idx = list(prices.columns).index(ticker)
            ax.plot(
                prices_normalized.index,
                prices_normalized[ticker],
                alpha=0.7,
                linewidth=1.5,
                color=colors[ticker_idx],
                label=ticker,
            )
        title_suffix = f"Representative Tickers ({n_tickers} selected)"

    # Plot average line (bold, black)
    ax.plot(
        avg_price.index,
        avg_price,
        label="VN30 Average",
        linewidth=3.5,
        color="black",
        linestyle="-",
        zorder=100,
    )

    ax.set_title(f"Price History - {title_suffix} (Normalized to 100)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Normalized Price (Base = 100)", fontsize=12)
    
    # Adjust legend for all 30 tickers
    if n_tickers is None or n_tickers >= len(prices.columns):
        # Place legend outside or use smaller font
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=3)
    else:
        ax.legend(loc="best", fontsize=9)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved price series plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_actual_prices(
    prices: pd.DataFrame,
    cfg: DictConfig,
    output_path: Path | str | None = None,
) -> None:
    """
    Plot actual closing prices (P_t) for all VN30 tickers without normalization.
    
    This shows the nominal price levels of each stock (e.g., 20k, 30k, 100k VND),
    not normalized returns or percentage changes.

    Args:
        prices: Prices DataFrame with actual closing prices
        cfg: Config
        output_path: Output path for figure
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    # Use colormap for colors
    cmap = plt.cm.get_cmap("tab20")
    colors = [cmap(i / len(prices.columns)) for i in range(len(prices.columns))]

    # Plot all 30 tickers with actual prices (no normalization)
    for idx, ticker in enumerate(prices.columns):
        ax.plot(
            prices.index,
            prices[ticker],
            alpha=0.7,
            linewidth=1.5,
            color=colors[idx],
            label=ticker,
        )

    ax.set_title("Actual Closing Prices - All 30 VN30 Tickers (Nominal VND)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Closing Price (VND)", fontsize=12)
    
    # Place legend outside
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=3)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved actual prices plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_distribution_daily_returns(
    rets_daily: pd.DataFrame,
    output_path: Path | str | None = None,
    sample_tickers: list[str] | None = None,
) -> None:
    """
    Plot histogram + KDE of daily log returns (all VN30 + sample tickers).

    Args:
        rets_daily: Daily log returns DataFrame
        output_path: Output path for figure
        sample_tickers: List of tickers to plot individually (default: VHM, VIC, HPG)
    """
    if sample_tickers is None:
        sample_tickers = ["VHM", "VIC", "HPG"]

    # Stack all returns for overall distribution
    all_returns = rets_daily.values.flatten()
    all_returns = all_returns[~np.isnan(all_returns)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Overall distribution (all VN30)
    ax = axes[0, 0]
    ax.hist(all_returns, bins=100, density=True, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=1, label="Zero return")
    ax.axvline(np.mean(all_returns), color="green", linestyle="--", linewidth=1, label="Mean")
    ax.set_title("Daily Log Returns Distribution - All VN30", fontsize=12)
    ax.set_xlabel("Log Return", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Individual tickers
    for idx, ticker in enumerate(sample_tickers[:3]):
        if ticker not in rets_daily.columns:
            continue
        row = (idx + 1) // 2
        col = (idx + 1) % 2
        ax = axes[row, col]

        ticker_returns = rets_daily[ticker].dropna()
        ax.hist(ticker_returns, bins=50, density=True, alpha=0.7, color="coral", edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.axvline(ticker_returns.mean(), color="green", linestyle="--", linewidth=1)
        ax.set_title(f"Daily Log Returns - {ticker}", fontsize=12)
        ax.set_xlabel("Log Return", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Daily Log Returns Distribution Analysis", fontsize=16, y=0.995)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved daily returns distribution to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_distribution_monthly_returns(
    rets_monthly: pd.DataFrame,
    output_path: Path | str | None = None,
) -> None:
    """
    Plot histogram of monthly simple returns (input to Markowitz).

    Args:
        rets_monthly: Monthly simple returns DataFrame
        output_path: Output path for figure
    """
    # Stack all monthly returns
    all_returns = rets_monthly.values.flatten()
    all_returns = all_returns[~np.isnan(all_returns)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall distribution
    ax = axes[0]
    ax.hist(all_returns, bins=50, density=True, alpha=0.7, color="darkgreen", edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero return")
    ax.axvline(np.mean(all_returns), color="blue", linestyle="--", linewidth=1.5, label="Mean")
    ax.set_title("Monthly Simple Returns Distribution - All VN30\n(Input to Markowitz)", fontsize=12)
    ax.set_xlabel("Simple Return", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot by ticker
    ax = axes[1]
    # Select subset of tickers for readability
    n_tickers_plot = min(15, len(rets_monthly.columns))
    selected = rets_monthly.iloc[:, :n_tickers_plot]
    data_to_plot = [selected[col].dropna().values for col in selected.columns]
    bp = ax.boxplot(data_to_plot, labels=selected.columns, vert=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title(f"Monthly Returns Distribution by Ticker\n(First {n_tickers_plot} tickers)", fontsize=12)
    ax.set_xlabel("Ticker", fontsize=10)
    ax.set_ylabel("Simple Return", fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Monthly Simple Returns Analysis (Markowitz Input)", fontsize=16, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved monthly returns distribution to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_correlation_heatmap(
    rets_monthly: pd.DataFrame,
    output_path: Path | str | None = None,
) -> None:
    """
    Plot correlation heatmap for VN30 monthly returns.

    Args:
        rets_monthly: Monthly simple returns DataFrame
        output_path: Output path for figure
    """
    corr = rets_monthly.corr()

    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(corr, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title("Correlation Matrix - VN30 Monthly Returns\n(Input to Markowitz)", fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", fontsize=10)

    # Add text annotations for high correlations
    threshold = 0.7
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            if i != j and abs(corr.iloc[i, j]) > threshold:
                text_color = "white" if abs(corr.iloc[i, j]) > 0.8 else "black"
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color=text_color, fontsize=6)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved correlation heatmap to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_cov_eigenvalues(
    rets_monthly: pd.DataFrame,
    output_path: Path | str | None = None,
) -> None:
    """
    Plot eigenvalues of covariance matrix (scree plot).

    Args:
        rets_monthly: Monthly simple returns DataFrame
        output_path: Output path for figure
    """
    # Compute covariance matrix
    cov = rets_monthly.cov()

    # Compute eigenvalues
    eigenvals = np.linalg.eigvals(cov)
    eigenvals = np.sort(eigenvals)[::-1]  # Sort descending

    # Compute explained variance
    total_var = eigenvals.sum()
    explained_var = eigenvals / total_var
    cumsum_var = np.cumsum(explained_var)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scree plot
    ax = axes[0]
    ax.plot(range(1, len(eigenvals) + 1), eigenvals, "o-", linewidth=2, markersize=6, color="steelblue")
    ax.set_title("Eigenvalues of Covariance Matrix\n(Scree Plot)", fontsize=12)
    ax.set_xlabel("Eigenvalue Index", fontsize=10)
    ax.set_ylabel("Eigenvalue", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)

    # Explained variance
    ax = axes[1]
    ax.plot(range(1, len(eigenvals) + 1), explained_var * 100, "o-", linewidth=2, markersize=6, color="darkgreen", label="Individual")
    ax.plot(range(1, len(eigenvals) + 1), cumsum_var * 100, "s-", linewidth=2, markersize=4, color="coral", label="Cumulative")
    ax.axhline(80, color="red", linestyle="--", linewidth=1, alpha=0.5, label="80% threshold")
    ax.set_title("Explained Variance by Eigenvalues", fontsize=12)
    ax.set_xlabel("Eigenvalue Index", fontsize=10)
    ax.set_ylabel("Explained Variance (%)", fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add text annotation for top eigenvalues
    n_top = min(5, len(eigenvals))
    top_var = cumsum_var[n_top - 1] * 100
    ax.text(n_top, top_var, f"Top {n_top}: {top_var:.1f}%", fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("Covariance Matrix Eigenvalue Analysis", fontsize=16, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved covariance eigenvalues plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_rolling_volatility(
    rets_daily: pd.DataFrame,
    window: int = 63,
    output_path: Path | str | None = None,
    sample_tickers: list[str] | None = None,
) -> None:
    """
    Plot rolling volatility over time for all VN30 tickers with colors.

    Args:
        rets_daily: Daily log returns DataFrame
        window: Rolling window size (default: 63 days ~ 3 months)
        output_path: Output path for figure
        sample_tickers: List of tickers to highlight (None = plot all 30)
    """
    # Compute rolling volatility (annualized)
    rolling_vol = rets_daily.rolling(window=window).std() * np.sqrt(252)

    # Average volatility across all VN30
    avg_vol = rolling_vol.mean(axis=1)

    # Min and max volatility bounds
    min_vol = rolling_vol.min(axis=1)
    max_vol = rolling_vol.max(axis=1)

    fig, ax = plt.subplots(figsize=(16, 8))

    # Use colormap for colors
    cmap = plt.cm.get_cmap("tab20")
    colors = [cmap(i / len(rolling_vol.columns)) for i in range(len(rolling_vol.columns))]

    # Plot all tickers with colors if showing all
    if sample_tickers is None:
        # Plot all 30 tickers with different colors
        for idx, ticker in enumerate(rolling_vol.columns):
            ticker_idx = list(rolling_vol.columns).index(ticker)
            ax.plot(
                rolling_vol.index,
                rolling_vol[ticker],
                alpha=0.5,
                linewidth=1.0,
                color=colors[ticker_idx],
                label=ticker,
            )
        # Plot min/max bounds
        ax.fill_between(
            rolling_vol.index,
            min_vol,
            max_vol,
            alpha=0.15,
            color="gray",
            label="Min-Max Range",
            zorder=1,
        )
    else:
        # Plot only sample tickers
        for ticker in sample_tickers:
            if ticker in rolling_vol.columns:
                ticker_idx = list(rolling_vol.columns).index(ticker)
                ax.plot(
                    rolling_vol.index,
                    rolling_vol[ticker],
                    label=ticker,
                    alpha=0.7,
                    linewidth=1.5,
                    color=colors[ticker_idx],
                )

    # Plot average (bold, always shown)
    ax.plot(
        rolling_vol.index,
        avg_vol,
        label="VN30 Average",
        linewidth=3.5,
        color="black",
        linestyle="-",
        zorder=100,
    )

    ax.set_title(
        f"Rolling Volatility ({window}-day window, annualized) - All VN30 Tickers",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Annualized Volatility", fontsize=12)
    
    # Adjust legend for all 30 tickers
    if sample_tickers is None:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=3)
    else:
        ax.legend(loc="best", fontsize=9)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved rolling volatility plot to {output_path}")
    else:
        plt.show()

    plt.close()
