import json
from types import SimpleNamespace

import pandas as pd

from src.vn50.plots import markowitz_results


def _build_cfg(tmp_path):
    cfg = SimpleNamespace()
    cfg.paths = SimpleNamespace()
    cfg.paths.reports = SimpleNamespace()
    cfg.paths.reports.artifacts = str(tmp_path / "reports" / "artifacts")
    cfg.paths.reports.logs = str(tmp_path / "reports" / "logs")
    return cfg


def test_load_backtest_results(tmp_path):
    cfg = _build_cfg(tmp_path)
    artifacts = tmp_path / "reports" / "artifacts"
    logs = tmp_path / "reports" / "logs"
    artifacts.mkdir(parents=True)
    logs.mkdir(parents=True)

    returns = pd.DataFrame(
        {
            "gmv": [0.01, -0.02],
            "tangency": [0.02, -0.01],
            "equal_weight": [0.015, -0.005],
        },
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )
    returns.to_csv(artifacts / "returns_walkforward_markowitz.csv", index_label="date")

    metrics = {
        "metrics": {
            "gmv": {"ann_return": 0.1},
            "tangency": {"ann_return": 0.12},
            "equal_weight": {"ann_return": 0.15},
        }
    }
    with open(logs / "metrics_walkforward_markowitz.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    df_loaded, metrics_loaded = markowitz_results.load_backtest_results(cfg)

    assert list(df_loaded.columns) == ["gmv", "tangency", "equal_weight"]
    assert pd.api.types.is_datetime64_any_dtype(df_loaded.index)
    assert set(metrics_loaded.keys()) == {"gmv", "tangency", "equal_weight"}

