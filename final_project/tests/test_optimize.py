"""Test portfolio optimization."""

import numpy as np
import pandas as pd
import pytest

from src.vn50.markowitz import estimators
from src.vn50.optimize import markowitz


def test_mean_variance_weights_gmv():
    """Test GMV optimization."""
    n_assets = 5
    returns = pd.DataFrame(
        np.random.randn(100, n_assets) * 0.01,
        columns=[f"Asset_{i}" for i in range(n_assets)],
        index=pd.date_range("2020-01-01", periods=100, freq="D"),
    )

    mu, Sigma = estimators.estimate_mu_sigma(returns, annualization_factor=252)

    weights_dict = markowitz.mean_variance_weights(
        mu, Sigma, objective="gmv", min_weight=0.0, max_weight=0.5
    )

    assert "gmv" in weights_dict
    w = weights_dict["gmv"]

    # Check constraints
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= 0).all()
    assert (w <= 0.5).all()


def test_mean_variance_weights_tangency():
    """Test Tangency portfolio optimization."""
    n_assets = 5
    returns = pd.DataFrame(
        np.random.randn(100, n_assets) * 0.01,
        columns=[f"Asset_{i}" for i in range(n_assets)],
        index=pd.date_range("2020-01-01", periods=100, freq="D"),
    )

    mu, Sigma = estimators.estimate_mu_sigma(returns, annualization_factor=252)

    weights_dict = markowitz.mean_variance_weights(
        mu, Sigma, objective="tangency", risk_free_rate=0.05, min_weight=0.0, max_weight=0.5
    )

    assert "tangency" in weights_dict
    w = weights_dict["tangency"]

    # Check constraints
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= 0).all()
    assert (w <= 0.5).all()

