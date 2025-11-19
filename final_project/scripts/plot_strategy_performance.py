from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_INPUT_PATH = Path("reports/logs/metrics_walkforward_markowitz.csv")
DEFAULT_OUTPUT_PATH = Path("reports/figures/strategy_performance_comparison.png")

STRATEGY_ORDER: Sequence[str] = ("gmv", "tangency", "equal_weight")
METRIC_COLUMNS: Sequence[str] = (
    "ann_return",
    "ann_vol",
    "sharpe",
    "max_drawdown",
    "calmar",
    "sortino",
    "turnover",
)

METRIC_LABELS: dict[str, str] = {
    "ann_return": "Lợi suất năm hóa (%)",
    "ann_vol": "Độ biến động năm hóa (%)",
    "sharpe": "Sharpe",
    "max_drawdown": "Max Drawdown (%)",
    "calmar": "Calmar",
    "sortino": "Sortino",
    "turnover": "Turnover (lần/năm)",
}

PERCENTAGE_COLUMNS: set[str] = {"ann_return", "ann_vol", "max_drawdown"}
BAR_COLORS: dict[str, str] = {
    "gmv": "#1f77b4",
    "tangency": "#ff7f0e",
    "equal_weight": "#2ca02c",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize backtest performance metrics for VN30 strategies."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"CSV file containing metrics (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to save the generated figure (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def format_display_value(metric: str, value: float) -> str:
    if metric in PERCENTAGE_COLUMNS:
        return f"{value * 100:.2f}%"
    if metric == "turnover":
        return f"{value:.2f}x"
    return f"{value:.2f}"


def main() -> None:
    arguments = parse_arguments()
    input_path: Path = arguments.input
    output_path: Path = arguments.output

    if not input_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {input_path}")

    metrics = pd.read_csv(input_path).set_index("strategy")
    missing_strategies = [strategy for strategy in STRATEGY_ORDER if strategy not in metrics.index]
    if missing_strategies:
        raise ValueError(f"Missing strategies in metrics file: {missing_strategies}")

    metrics = metrics.loc[STRATEGY_ORDER, METRIC_COLUMNS]

    num_metrics = len(METRIC_COLUMNS)
    num_rows = 4
    num_cols = 2
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(14, 12),
        constrained_layout=True,
    )

    axes_flat = axes.flatten()
    for metric_index, metric_name in enumerate(METRIC_COLUMNS):
        axis = axes_flat[metric_index]
        series = metrics[metric_name]
        values = series.copy()
        if metric_name in PERCENTAGE_COLUMNS:
            values = values * 100

        x_positions = range(len(series.index))
        bars = axis.bar(
            x_positions,
            values,
            color=[BAR_COLORS[strategy] for strategy in series.index],
        )
        axis.set_title(METRIC_LABELS[metric_name], fontsize=11, fontweight="bold")
        axis.grid(axis="y", linestyle="--", alpha=0.4)
        axis.set_xticks(
            list(x_positions),
            [strategy.upper() for strategy in series.index],
        )

        for bar, original_value in zip(bars, series.values, strict=True):
            axis.annotate(
                format_display_value(metric_name, original_value),
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Remove unused subplot if metrics < rows*cols
    for leftover_axis in axes_flat[num_metrics:]:
        leftover_axis.axis("off")

    fig.suptitle(
        "Hiệu suất backtest các chiến lược VN30 (2020-10 đến 2025-10)",
        fontsize=14,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

