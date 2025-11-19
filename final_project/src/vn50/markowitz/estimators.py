"""Estimate parameters for Markowitz optimization: mu and Sigma."""

import logging

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError

logger = logging.getLogger(__name__)


def estimate_mu_sigma(
    returns: pd.DataFrame,
    shrinkage: bool = True,
    shrinkage_factor: float = 0.2,
    jitter: float = 1e-6,
    annualization_factor: int = 12,  # 12 for monthly returns
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Estimate annualized mean returns (mu) and covariance matrix (Sigma).

    Args:
        returns: DataFrame with returns (monthly for Markowitz)
        shrinkage: Apply shrinkage to covariance
        shrinkage_factor: Shrinkage factor (0-1)
        jitter: Small value added to diagonal for numerical stability
        annualization_factor: Factor to annualize (12 for monthly, 252 for daily)

    Returns:
        Tuple of (mu: pd.Series, Sigma: pd.DataFrame)
        Both are annualized and aligned by ticker.
    """
    # Drop any remaining NaN
    returns_clean = returns.dropna()

    if len(returns_clean) == 0:
        raise ValueError("No valid returns data after dropping NaN")

    # Estimate mean returns (annualized)
    mu = returns_clean.mean() * annualization_factor
    mu = mu.sort_index()

    # Estimate covariance matrix
    sample_cov = returns_clean.cov() * annualization_factor

    # Apply shrinkage if requested
    if shrinkage:
        # Shrink towards diagonal matrix (constant correlation model)
        n_assets = len(sample_cov)
        avg_var = np.diag(sample_cov).mean()
        target = np.eye(n_assets) * avg_var

        Sigma = (1 - shrinkage_factor) * sample_cov + shrinkage_factor * target
        logger.info(f"Applied shrinkage (factor={shrinkage_factor}) to covariance")
    else:
        Sigma = sample_cov

    # Add jitter to diagonal for numerical stability
    Sigma = Sigma.copy()
    np.fill_diagonal(Sigma.values, Sigma.values.diagonal() + jitter)

    # Ensure PSD (positive semi-definite)
    try:
        # Check if already PSD
        eigenvals = np.linalg.eigvals(Sigma.values)
        if np.any(eigenvals < -1e-8):
            logger.warning("Covariance matrix not PSD, fixing...")
            # Make PSD by setting negative eigenvalues to small positive value
            eigenvals = np.maximum(eigenvals, 1e-8)
            eigenvecs = np.linalg.eig(Sigma.values)[1]
            Sigma = pd.DataFrame(
                eigenvecs @ np.diag(eigenvals) @ eigenvecs.T,
                index=Sigma.index,
                columns=Sigma.columns,
            )
    except LinAlgError as e:
        logger.error(f"Error checking PSD: {e}")
        raise

    # Align mu and Sigma
    common_tickers = mu.index.intersection(Sigma.index)
    mu = mu.loc[common_tickers]
    Sigma = Sigma.loc[common_tickers, common_tickers]

    logger.info(
        f"Estimated mu and Sigma: {len(mu)} assets, "
        f"min eigenval={np.linalg.eigvals(Sigma.values).min():.2e}"
    )

    return mu, Sigma

