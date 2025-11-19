"""Efficient frontier plotting utilities for VN30 Markowitz."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.vn50.markowitz import estimators
from src.vn50.optimize import markowitz

logger = logging.getLogger(__name__)


def load_monthly_returns_for_frontier(cfg: DictConfig) -> pd.DataFrame:
    """Load monthly returns cleaned for efficient frontier estimation."""
    processed_dir = Path(cfg.paths.data.processed)
    parquet_path = processed_dir / "returns_monthly_simple.parquet"
    csv_path = processed_dir / "returns_monthly_simple.csv"

    if parquet_path.exists():
        returns = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        returns = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        raise FileNotFoundError("returns_monthly_simple file not found. Run preprocess pipeline first.")

    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)

    if cfg.data.get("universe"):
        universe_path = Path(cfg.data.universe)
        universe_cfg = OmegaConf.load(universe_path)
        tickers = universe_cfg.get("tickers", universe_cfg)
        returns = returns.loc[:, returns.columns.intersection(tickers)]

    returns = returns.dropna(axis=1, how="all").dropna(axis=0, how="all").sort_index()
    logger.info("Loaded monthly returns for %d assets (%d observations)", returns.shape[1], returns.shape[0])
    return returns


def compute_mu_sigma_from_cfg(cfg: DictConfig) -> tuple[pd.Series, pd.DataFrame]:
    """Estimate mu and Sigma using config parameters."""
    returns = load_monthly_returns_for_frontier(cfg)
    mu, Sigma = estimators.estimate_mu_sigma(
        returns,
        shrinkage=cfg.features.cov.shrinkage,
        shrinkage_factor=cfg.features.cov.shrinkage_factor,
        jitter=cfg.features.cov.jitter,
        annualization_factor=12,
    )
    return mu, Sigma


def _portfolio_stats(weights: pd.Series, mu: pd.Series, Sigma: pd.DataFrame) -> tuple[float, float]:
    aligned = weights.index.intersection(mu.index)
    w = weights.loc[aligned].values
    mu_vec = mu.loc[aligned].values
    Sigma_mat = Sigma.loc[aligned, aligned].values
    ret = float(np.dot(w, mu_vec))
    vol = float(np.sqrt(w @ Sigma_mat @ w))
    return ret, vol


def plot_efficient_frontier_static(mu: pd.Series, Sigma: pd.DataFrame, cfg: DictConfig, out_dir: Path) -> Path:
    """Plot efficient frontier with GMV/Tangency/EW points."""
    frontier = markowitz.compute_efficient_frontier(
        mu,
        Sigma,
        n_points=50,
        min_weight=cfg.markowitz_manual.min_weight,
        max_weight=cfg.markowitz_manual.max_weight,
        risk_free_rate=cfg.markowitz_manual.risk_free_rate,
        solver=cfg.markowitz_manual.solver,
        max_iter=cfg.markowitz_manual.max_iter,
    )

    weights_dict = markowitz.mean_variance_weights(
        mu,
        Sigma,
        objective="frontier",
        risk_free_rate=cfg.markowitz_manual.risk_free_rate,
        min_weight=cfg.markowitz_manual.min_weight,
        max_weight=cfg.markowitz_manual.max_weight,
        solver=cfg.markowitz_manual.solver,
        max_iter=cfg.markowitz_manual.max_iter,
    )

    overlay_points = {}
    if weights_dict.get("gmv") is not None:
        ret, vol = _portfolio_stats(weights_dict["gmv"], mu, Sigma)
        overlay_points["GMV"] = (vol * 100, ret * 100)
    if weights_dict.get("tangency") is not None:
        ret, vol = _portfolio_stats(weights_dict["tangency"], mu, Sigma)
        overlay_points["Tangency"] = (vol * 100, ret * 100)

    w_equal = pd.Series(1.0 / len(mu), index=mu.index)
    ret_equal, vol_equal = _portfolio_stats(w_equal, mu, Sigma)
    overlay_points["Equal-weight"] = (vol_equal * 100, ret_equal * 100)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(frontier["ann_vol_pct"], frontier["ann_return_pct"], "-o", markersize=3, label="Frontier")

    markers = {"GMV": "s", "Tangency": "D", "Equal-weight": "o"}
    colors = {"GMV": "#C44E52", "Tangency": "#55A868", "Equal-weight": "#4C72B0"}

    for name, (vol, ret) in overlay_points.items():
        ax.scatter(vol, ret, label=name, marker=markers.get(name, "o"), s=80, color=colors.get(name, "#000000"))
        ax.annotate(name, (vol, ret), textcoords="offset points", xytext=(5, 5))

    ax.set_title("Đường biên hiệu quả Markowitz – VN30 (monthly returns)")
    ax.set_xlabel("Độ biến động năm hóa (%)")
    ax.set_ylabel("Lợi suất năm hóa kỳ vọng (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "efficient_frontier_vn30_markowitz.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved efficient frontier plot to %s", output_path)
    return output_path

