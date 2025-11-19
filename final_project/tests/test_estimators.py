"""Test parameter estimation."""

import numpy as np
import pandas as pd
import pytest

from src.vn50.markowitz import estimators


def test_estimate_mu_sigma():
    """Test mu and Sigma estimation."""
    n_assets = 5
    returns = pd.DataFrame(
        np.random.randn(100, n_assets) * 0.01,
        columns=[f"Asset_{i}" for i in range(n_assets)],
        index=pd.date_range("2020-01-01", periods=100, freq="D"),
    )

    mu, Sigma = estimators.estimate_mu_sigma(returns, annualization_factor=252)

    assert len(mu) == n_assets
    assert Sigma.shape == (n_assets, n_assets)
    assert mu.index.equals(Sigma.index)
    assert mu.index.equals(Sigma.columns)

    # Check PSD
    eigenvals = np.linalg.eigvals(Sigma.values)
    assert (eigenvals >= -1e-8).all()


def test_estimate_mu_sigma_shrinkage():
    """Test shrinkage."""
    n_assets = 5
    returns = pd.DataFrame(
        np.random.randn(100, n_assets) * 0.01,
        columns=[f"Asset_{i}" for i in range(n_assets)],
        index=pd.date_range("2020-01-01", periods=100, freq="D"),
    )

    mu, Sigma = estimators.estimate_mu_sigma(
        returns, shrinkage=True, shrinkage_factor=0.2, annualization_factor=252
    )

    # Check PSD
    eigenvals = np.linalg.eigvals(Sigma.values)
    assert (eigenvals >= -1e-8).all()

