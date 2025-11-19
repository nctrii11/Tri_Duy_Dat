import json
from pathlib import Path

import pandas as pd
import pytest

from src.vn50.analysis import benchmark_markowitz


def test_load_markowitz_metrics(tmp_path):
    metrics_data = {
        "metrics": {
            "gmv": {"ann_return": 0.1, "ann_vol": 0.12, "sharpe": 0.5, "max_drawdown": -0.05, "calmar": 2.0, "sortino": 0.7, "turnover": 0.3},
            "tangency": {"ann_return": 0.12, "ann_vol": 0.15, "sharpe": 0.4, "max_drawdown": -0.08, "calmar": 1.5, "sortino": 0.6, "turnover": 0.8},
            "equal_weight": {"ann_return": 0.09, "ann_vol": 0.13, "sharpe": 0.45, "max_drawdown": -0.04, "calmar": 2.2, "sortino": 0.65, "turnover": 0.0},
        }
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_data))

    metrics = benchmark_markowitz.load_markowitz_metrics(metrics_path)

    assert set(metrics.keys()) == {"gmv", "tangency", "equal_weight"}


def test_compute_relative_to_benchmark():
    metrics = {
        "gmv": {"ann_return": 0.1, "ann_vol": 0.12, "sharpe": 0.5, "max_drawdown": -0.05},
        "equal_weight": {"ann_return": 0.09, "ann_vol": 0.13, "sharpe": 0.45, "max_drawdown": -0.04},
    }
    rel = benchmark_markowitz.compute_relative_to_benchmark(metrics, benchmark_key="equal_weight")

    assert "gmv" in rel
    assert rel["gmv"]["excess_ann_return"] == pytest.approx(0.01)
    assert rel["gmv"]["vol_ratio"] == pytest.approx(metrics["gmv"]["ann_vol"] / metrics["equal_weight"]["ann_vol"])


def test_build_benchmark_table_and_markdown():
    metrics = {
        "gmv": {"ann_return": 0.1, "ann_vol": 0.12, "sharpe": 0.5, "max_drawdown": -0.05, "calmar": 2.0, "sortino": 0.7, "turnover": 0.3},
        "tangency": {"ann_return": 0.12, "ann_vol": 0.15, "sharpe": 0.4, "max_drawdown": -0.08, "calmar": 1.5, "sortino": 0.6, "turnover": 0.8},
        "equal_weight": {"ann_return": 0.09, "ann_vol": 0.13, "sharpe": 0.45, "max_drawdown": -0.04, "calmar": 2.2, "sortino": 0.65, "turnover": 0.0},
    }
    rel = benchmark_markowitz.compute_relative_to_benchmark(metrics, benchmark_key="equal_weight")
    table = benchmark_markowitz.build_benchmark_table(metrics, rel)

    assert "GMV" in table.index[0] or "EQUAL-WEIGHT" in table.index
    assert "ann_return_pct" in table.columns

    markdown = benchmark_markowitz.render_benchmark_markdown(table, rel)
    assert "Benchmark Equal-weight" in markdown
    assert "GMV" in markdown

