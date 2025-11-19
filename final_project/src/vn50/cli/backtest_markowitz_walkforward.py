"""CLI: Walk-forward backtest for Markowitz portfolio optimization."""

import json
import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.vn50.backtest.walkforward import plot_nav_curves, walk_forward_backtest
from src.vn50.markowitz import estimators
from src.vn50.optimize import markowitz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main_app(cfg: DictConfig) -> None:
    """Run walk-forward backtest for Markowitz portfolio."""
    logger.info(f"Walk-forward backtest for experiment: {cfg.experiment}")

    # 1) Load monthly returns
    processed_path = Path(cfg.paths.data.processed)
    monthly_returns_path = processed_path / "returns_monthly_simple.csv"

    if not monthly_returns_path.exists():
        # Try parquet
        monthly_returns_path = processed_path / "returns_monthly_simple.parquet"
        if monthly_returns_path.exists():
            rets_m = pd.read_parquet(monthly_returns_path)
        else:
            raise FileNotFoundError(
                f"Monthly returns file not found: {processed_path}/returns_monthly_simple.csv or .parquet"
            )
    else:
        rets_m = pd.read_csv(monthly_returns_path, index_col=0, parse_dates=True)

    rets_m = rets_m.sort_index()
    logger.info(f"Loaded monthly returns: {len(rets_m)} periods, {len(rets_m.columns)} assets")

    # 2) Định nghĩa solver Markowitz
    def solver_fn(mu, Sigma, cfg_):
        """Wrapper cho mean_variance_weights - trả về cả GMV và Tangency."""
        result = {}
        
        # GMV
        gmv_result = markowitz.mean_variance_weights(
            mu,
            Sigma,
            objective="gmv",
            risk_free_rate=cfg_.markowitz_manual.risk_free_rate,
            min_weight=cfg_.markowitz_manual.min_weight,
            max_weight=cfg_.markowitz_manual.max_weight,
            solver=cfg_.markowitz_manual.solver,
            max_iter=cfg_.markowitz_manual.max_iter,
        )
        if "gmv" in gmv_result:
            result["gmv"] = gmv_result["gmv"]
        
        # Tangency
        tangency_result = markowitz.mean_variance_weights(
            mu,
            Sigma,
            objective="tangency",
            risk_free_rate=cfg_.markowitz_manual.risk_free_rate,
            min_weight=cfg_.markowitz_manual.min_weight,
            max_weight=cfg_.markowitz_manual.max_weight,
            solver=cfg_.markowitz_manual.solver,
            max_iter=cfg_.markowitz_manual.max_iter,
        )
        if "tangency" in tangency_result:
            result["tangency"] = tangency_result["tangency"]
        
        return result

    # 3) Gọi walk_forward_backtest
    logger.info("Starting walk-forward backtest...")
    result = walk_forward_backtest(
        returns_monthly=rets_m,
        cfg=cfg,
        markowitz_solver_fn=solver_fn,
    )

    # 4) Lưu kết quả
    out_dir_logs = Path(cfg.paths.reports.logs)
    out_dir_artifacts = Path(cfg.paths.reports.artifacts)
    out_dir_logs.mkdir(parents=True, exist_ok=True)
    out_dir_artifacts.mkdir(parents=True, exist_ok=True)

    # NAV DataFrame
    nav_dict = {name: strat.nav for name, strat in result.strategies.items()}
    nav_df = pd.DataFrame(nav_dict)
    nav_output_path = out_dir_artifacts / "nav_walkforward_markowitz.csv"
    nav_df.to_csv(nav_output_path)
    logger.info(f"Saved NAV to {nav_output_path}")

    # Returns DataFrame
    returns_dict = {name: strat.returns for name, strat in result.strategies.items()}
    returns_df = pd.DataFrame(returns_dict)
    returns_output_path = out_dir_artifacts / "returns_walkforward_markowitz.csv"
    returns_df.to_csv(returns_output_path)
    logger.info(f"Saved returns to {returns_output_path}")

    # Metrics
    metrics_rows = []
    for name, strat in result.strategies.items():
        row = {"strategy": name}
        row.update(strat.metrics)
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_output_path = out_dir_logs / "metrics_walkforward_markowitz.csv"
    metrics_df.to_csv(metrics_output_path, index=False)
    logger.info(f"Saved metrics to {metrics_output_path}")

    # Metrics JSON
    metrics_json = {name: strat.metrics for name, strat in result.strategies.items()}
    metrics_json_path = out_dir_logs / "metrics_walkforward_markowitz.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    logger.info(f"Saved metrics JSON to {metrics_json_path}")

    # Log summary
    logger.info("=" * 60)
    logger.info("Walk-forward backtest completed!")
    logger.info("=" * 60)
    logger.info("Metrics summary:")
    for name, strat in result.strategies.items():
        logger.info(f"\n{name.upper()}:")
        for metric_name, metric_value in strat.metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")

    # 5) Vẽ biểu đồ NAV
    nav_plot_path = out_dir_artifacts / "nav_walkforward_markowitz.png"
    plot_nav_curves(result, output_path=nav_plot_path)
    logger.info(f"Saved NAV plot to {nav_plot_path}")


if __name__ == "__main__":
    main_app()

