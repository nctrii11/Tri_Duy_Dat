"""Markowitz portfolio optimization: manual (scipy/cvxpy) and PyPortfolioOpt."""

import logging
from typing import Literal

import cvxpy as cp
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, objective_functions, risk_models
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def mean_variance_weights(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    objective: Literal["gmv", "tangency", "frontier"] = "tangency",
    risk_free_rate: float = 0.05,
    min_weight: float = 0.0,
    max_weight: float = 0.20,
    solver: Literal["SLSQP", "CVXPY"] = "SLSQP",
    max_iter: int = 1000,
) -> dict[str, pd.Series | pd.DataFrame]:
    """
    Calculate Markowitz portfolio weights (manual implementation).

    Args:
        mu: Annualized mean returns (pd.Series)
        Sigma: Annualized covariance matrix (pd.DataFrame)
        objective: "gmv" (Global Minimum Variance), "tangency" (max Sharpe), "frontier"
        risk_free_rate: Annualized risk-free rate
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset
        solver: "SLSQP" or "CVXPY"
        max_iter: Maximum iterations

    Returns:
        Dictionary with weights:
        - "gmv": GMV weights (if objective includes gmv)
        - "tangency": Tangency portfolio weights (if objective includes tangency)
        - "frontier": DataFrame with efficient frontier points (if objective="frontier")
    """
    # Align mu and Sigma
    common_tickers = mu.index.intersection(Sigma.index)
    mu_aligned = mu.loc[common_tickers]
    Sigma_aligned = Sigma.loc[common_tickers, common_tickers]

    n_assets = len(mu_aligned)
    mu_array = mu_aligned.values
    Sigma_array = Sigma_aligned.values

    result = {}

    # GMV (Global Minimum Variance)
    if objective in ["gmv", "frontier"]:
        w_gmv = _solve_gmv(
            Sigma_array, min_weight, max_weight, solver, max_iter
        )
        result["gmv"] = pd.Series(w_gmv, index=common_tickers, name="GMV")

    # Tangency (Max Sharpe)
    if objective in ["tangency", "frontier"]:
        w_tangency = _solve_tangency(
            mu_array,
            Sigma_array,
            risk_free_rate,
            min_weight,
            max_weight,
            solver,
            max_iter,
        )
        result["tangency"] = pd.Series(
            w_tangency, index=common_tickers, name="Tangency"
        )

    # Efficient Frontier
    if objective == "frontier":
        frontier_weights = _solve_efficient_frontier(
            mu_array,
            Sigma_array,
            min_weight,
            max_weight,
            solver,
            max_iter,
            n_points=50,
        )
        result["frontier"] = pd.DataFrame(
            frontier_weights, index=common_tickers
        ).T

    return result


def _solve_gmv(
    Sigma: np.ndarray,
    min_weight: float,
    max_weight: float,
    solver: str,
    max_iter: int,
) -> np.ndarray:
    """Solve Global Minimum Variance portfolio."""
    n = len(Sigma)

    if solver == "SLSQP":
        # Objective: minimize w^T Sigma w
        def objective(w):
            return w @ Sigma @ w

        # Constraints: sum(w) = 1, min_weight <= w <= max_weight
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(min_weight, max_weight) for _ in range(n)]
        x0 = np.ones(n) / n

        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": max_iter}
        )

        if not result.success:
            logger.warning(f"GMV optimization may not have converged: {result.message}")

        return result.x

    elif solver == "CVXPY":
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, Sigma))
        constraints = [
            cp.sum(w) == 1,
            w >= min_weight,
            w <= max_weight,
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status != "optimal":
            logger.warning(f"GMV CVXPY status: {problem.status}")

        return w.value

    else:
        raise ValueError(f"Unknown solver: {solver}")


def _solve_tangency(
    mu: np.ndarray,
    Sigma: np.ndarray,
    risk_free_rate: float,
    min_weight: float,
    max_weight: float,
    solver: str,
    max_iter: int,
) -> np.ndarray:
    """Solve Tangency portfolio (max Sharpe ratio)."""
    n = len(mu)
    excess_mu = mu - risk_free_rate

    if solver == "SLSQP":
        # Objective: maximize (w^T mu - rf) / sqrt(w^T Sigma w)
        # Equivalent to minimize -Sharpe
        def objective(w):
            portfolio_return = w @ mu
            portfolio_vol = np.sqrt(w @ Sigma @ w)
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            return -sharpe

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(min_weight, max_weight) for _ in range(n)]
        x0 = np.ones(n) / n

        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": max_iter}
        )

        if not result.success:
            logger.warning(
                f"Tangency optimization may not have converged: {result.message}"
            )

        return result.x

    elif solver == "CVXPY":
        w = cp.Variable(n)
        portfolio_return = mu @ w
        portfolio_vol = cp.quad_form(w, Sigma) ** 0.5
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
        objective = cp.Maximize(sharpe)
        constraints = [
            cp.sum(w) == 1,
            w >= min_weight,
            w <= max_weight,
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status != "optimal":
            logger.warning(f"Tangency CVXPY status: {problem.status}")

        return w.value

    else:
        raise ValueError(f"Unknown solver: {solver}")


def _solve_efficient_frontier(
    mu: np.ndarray,
    Sigma: np.ndarray,
    min_weight: float,
    max_weight: float,
    solver: str,
    max_iter: int,
    n_points: int = 50,
) -> np.ndarray:
    """Solve efficient frontier (multiple risk-return points)."""
    n = len(mu)

    # Find min and max expected returns
    w_min_vol = _solve_gmv(Sigma, min_weight, max_weight, solver, max_iter)
    min_return = mu @ w_min_vol

    # Max return portfolio (subject to constraints)
    w_max_return = _solve_max_return(mu, min_weight, max_weight, solver, max_iter)
    max_return = mu @ w_max_return

    # Generate target returns
    target_returns = np.linspace(min_return, max_return, n_points)

    frontier_weights = []
    for target_ret in target_returns:
        w = _solve_min_vol_for_return(
            mu, Sigma, target_ret, min_weight, max_weight, solver, max_iter
        )
        frontier_weights.append(w)

    return np.array(frontier_weights).T


def _solve_max_return(
    mu: np.ndarray,
    min_weight: float,
    max_weight: float,
    solver: str,
    max_iter: int,
) -> np.ndarray:
    """Solve maximum return portfolio."""
    n = len(mu)

    if solver == "SLSQP":
        def objective(w):
            return -(mu @ w)  # Minimize negative return

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(min_weight, max_weight) for _ in range(n)]
        x0 = np.ones(n) / n

        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": max_iter}
        )
        return result.x

    elif solver == "CVXPY":
        w = cp.Variable(n)
        objective = cp.Maximize(mu @ w)
        constraints = [
            cp.sum(w) == 1,
            w >= min_weight,
            w <= max_weight,
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return w.value

    else:
        raise ValueError(f"Unknown solver: {solver}")


def _solve_min_vol_for_return(
    mu: np.ndarray,
    Sigma: np.ndarray,
    target_return: float,
    min_weight: float,
    max_weight: float,
    solver: str,
    max_iter: int,
) -> np.ndarray:
    """Solve minimum volatility portfolio for given target return."""
    n = len(mu)

    if solver == "SLSQP":
        def objective(w):
            return w @ Sigma @ w

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: mu @ w - target_return},
        ]
        bounds = [(min_weight, max_weight) for _ in range(n)]
        x0 = np.ones(n) / n

        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": max_iter}
        )
        return result.x

    elif solver == "CVXPY":
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, Sigma))
        constraints = [
            cp.sum(w) == 1,
            mu @ w == target_return,
            w >= min_weight,
            w <= max_weight,
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return w.value

    else:
        raise ValueError(f"Unknown solver: {solver}")


def compute_efficient_frontier(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    n_points: int = 50,
    min_weight: float = 0.0,
    max_weight: float = 0.20,
    risk_free_rate: float | None = None,
    solver: Literal["SLSQP", "CVXPY"] = "SLSQP",
    max_iter: int = 1000,
) -> pd.DataFrame:
    """Compute efficient frontier risk-return pairs."""
    if n_points <= 1:
        raise ValueError("n_points must be greater than 1")

    common_tickers = mu.index.intersection(Sigma.index)
    mu_aligned = mu.loc[common_tickers]
    Sigma_aligned = Sigma.loc[common_tickers, common_tickers]

    mu_array = mu_aligned.values
    Sigma_array = Sigma_aligned.values

    w_min = _solve_gmv(Sigma_array, min_weight, max_weight, solver, max_iter)
    min_return = float(mu_array @ w_min)
    w_max = _solve_max_return(mu_array, min_weight, max_weight, solver, max_iter)
    max_return = float(mu_array @ w_max)

    if np.isclose(min_return, max_return):
        target_returns = np.array([min_return])
    else:
        target_returns = np.linspace(min_return, max_return, n_points)

    records = []
    weights_col = []

    for target_ret in target_returns:
        try:
            weights = _solve_min_vol_for_return(
                mu_array,
                Sigma_array,
                target_ret,
                min_weight,
                max_weight,
                solver,
                max_iter,
            )
        except ValueError:
            continue
        weights_series = pd.Series(weights, index=common_tickers)
        port_return = float((weights_series * mu_aligned).sum())
        port_vol = float(np.sqrt(weights_series.values @ Sigma_array @ weights_series.values))
        sharpe = (
            (port_return - risk_free_rate) / port_vol
            if risk_free_rate is not None and port_vol > 0
            else np.nan
        )
        records.append(
            {
                "ann_return_pct": port_return * 100,
                "ann_vol_pct": port_vol * 100,
                "sharpe": sharpe,
            }
        )
        weights_col.append(weights_series)

    frontier_df = pd.DataFrame(records)
    frontier_df["weights"] = weights_col
    return frontier_df


def optimize_markowitz_pypfopt(
    returns: pd.DataFrame,
    risk_model: str = "sample_cov",
    objective: Literal["min_volatility", "max_sharpe"] = "max_sharpe",
    risk_free_rate: float = 0.05,
    weight_bounds: tuple[float, float] = (0.0, 0.20),
    L2_penalty: float = 0.0,
) -> dict[str, pd.Series]:
    """
    Optimize portfolio using PyPortfolioOpt.

    Args:
        returns: DataFrame with returns
        risk_model: Risk model ("sample_cov", "ledoit_wolf", etc.)
        objective: "min_volatility" or "max_sharpe"
        risk_free_rate: Annualized risk-free rate
        weight_bounds: (min_weight, max_weight)
        L2_penalty: L2 regularization penalty

    Returns:
        Dictionary with weights: {"gmv": ..., "tangency": ...}
    """
    # Estimate expected returns and covariance
    mu = expected_returns.mean_historical_return(
        returns,
        returns_data=True,
        frequency=12,
    )
    if risk_model == "sample_cov":
        S = risk_models.sample_cov(
            returns,
            returns_data=True,
            frequency=12,
        )
    elif risk_model == "ledoit_wolf":
        S = risk_models.CovarianceShrinkage(
            returns,
            returns_data=True,
            frequency=12,
        ).ledoit_wolf()
    else:
        raise ValueError(f"Unknown risk_model: {risk_model}")

    result = {}

    # Create EfficientFrontier
    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)

    # GMV (min volatility)
    ef_min_vol = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    if L2_penalty > 0:
        ef_min_vol.add_objective(
            objective_functions.L2_reg, gamma=L2_penalty
        )
    w_gmv_dict = ef_min_vol.min_volatility()
    result["gmv"] = pd.Series(w_gmv_dict, name="GMV")

    # Tangency (max Sharpe)
    ef_sharpe = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    if L2_penalty > 0:
        ef_sharpe.add_objective(
            objective_functions.L2_reg, gamma=L2_penalty
        )
    w_tangency_dict = ef_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
    result["tangency"] = pd.Series(w_tangency_dict, name="Tangency")

    logger.info(f"PyPortfolioOpt optimization completed: {list(result.keys())}")

    return result

