"""Benchmark analysis utilities for Markowitz walk-forward results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_markowitz_metrics(metrics_path: Path) -> dict:
    """Load metrics JSON produced by run_markowitz CLI."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    metrics = payload.get("metrics", {})
    if not metrics:
        raise ValueError(f"No metrics found in {metrics_path}")

    logger.info("Loaded metrics for strategies: %s", ", ".join(metrics.keys()))
    return metrics


def compute_relative_to_benchmark(metrics: dict, benchmark_key: str = "equal_weight") -> dict:
    """Compute relative performance statistics vs benchmark."""
    if benchmark_key not in metrics:
        raise KeyError(f"Benchmark '{benchmark_key}' not found in metrics.")

    benchmark = metrics[benchmark_key]
    rel = {"benchmark_name": benchmark_key}
    for name, vals in metrics.items():
        if name == benchmark_key:
            continue
        rel[name] = {
            "excess_ann_return": vals.get("ann_return", 0.0) - benchmark.get("ann_return", 0.0),
            "excess_sharpe": vals.get("sharpe", 0.0) - benchmark.get("sharpe", 0.0),
            "vol_ratio": vals.get("ann_vol", 1.0) / benchmark.get("ann_vol", 1.0)
            if benchmark.get("ann_vol", 0.0) not in (None, 0.0)
            else float("nan"),
            "dd_diff": vals.get("max_drawdown", 0.0) - benchmark.get("max_drawdown", 0.0),
        }
    return rel


def build_benchmark_table(metrics: dict, rel_metrics: dict) -> pd.DataFrame:
    """Build combined table with absolute and relative metrics."""
    records = []
    for name, vals in metrics.items():
        record = {
            "strategy": name.upper() if name != rel_metrics.get("benchmark_name") else "EQUAL-WEIGHT (BENCHMARK)",
            "ann_return_pct": vals.get("ann_return", float("nan")) * 100,
            "ann_vol_pct": vals.get("ann_vol", float("nan")) * 100,
            "sharpe": vals.get("sharpe", float("nan")),
            "max_drawdown_pct": vals.get("max_drawdown", float("nan")) * 100,
            "calmar": vals.get("calmar", float("nan")),
            "sortino": vals.get("sortino", float("nan")),
            "turnover": vals.get("turnover", float("nan")),
            "excess_ann_return_pct": None,
            "excess_sharpe": None,
            "vol_ratio": None,
        }
        rel = rel_metrics.get(name)
        if rel:
            record["excess_ann_return_pct"] = rel["excess_ann_return"] * 100
            record["excess_sharpe"] = rel["excess_sharpe"]
            record["vol_ratio"] = rel["vol_ratio"]
        records.append(record)

    table = pd.DataFrame(records)
    return table.set_index("strategy")


def render_benchmark_markdown(table: pd.DataFrame, rel_metrics: dict) -> str:
    """Render Markdown summary for benchmark comparison."""
    md_lines = []
    md_lines.append("## So sánh Markowitz với Benchmark Equal-weight\n")
    md_lines.append(
        "Phần này đánh giá hai chiến lược Markowitz (GMV và Tangency) so với danh mục "
        "Equal-weight (EW) áp dụng cho rổ VN30 trong backtest walk-forward 36 tháng."
    )
    md_lines.append("")
    try:
        table_md = table.to_markdown(tablefmt="github", floatfmt=".2f")
    except ImportError:
        table_md = table.to_string()
        logger.warning("tabulate not installed; falling back to plain text table.")
    md_lines.append(table_md)
    md_lines.append("")

    benchmark_name = rel_metrics.get("benchmark_name", "equal_weight")
    gmv_rel = rel_metrics.get("gmv", {})
    tan_rel = rel_metrics.get("tangency", {})

    if gmv_rel:
        md_lines.append(
            f"- **GMV** đạt excess return khoảng {gmv_rel.get('excess_ann_return', 0.0) * 100:.2f} điểm % "
            f"so với {benchmark_name.upper()}, Sharpe chênh {gmv_rel.get('excess_sharpe', 0.0):.2f}. "
            f"Tỷ lệ biến động (vol ratio) {gmv_rel.get('vol_ratio', float('nan')):.2f}, cho thấy mức rủi ro "
            f"thấp hơn nếu <1."
        )
    if tan_rel:
        md_lines.append(
            f"- **Tangency** có excess return {tan_rel.get('excess_ann_return', 0.0) * 100:.2f} điểm %, "
            f"Sharpe chênh {tan_rel.get('excess_sharpe', 0.0):.2f}, vol ratio {tan_rel.get('vol_ratio', float('nan')):.2f}. "
            "Chiến lược này ưu tiên lợi suất nên biến động cao hơn benchmark."
        )
    md_lines.append(
        "Equal-weight đóng vai trò chuẩn tham chiếu; cả hai chiến lược Markowitz đều được so sánh trực tiếp "
        "trên cùng dữ liệu tháng ngoài mẫu, đảm bảo không look-ahead."
    )

    return "\n".join(md_lines)

