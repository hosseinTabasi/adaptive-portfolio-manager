"""Backtest risk diagnostics: drawdown, historical VaR / CVaR.

These are sample statistics of a simulated path. They are not regulatory
capital numbers and they are not forecasts of future loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RiskReport:
    var_95: float  # positive number: loss quantile of simple returns
    cvar_95: float
    var_95_ann_approx: float
    n_obs: int
    method: str


def max_drawdown(equity: pd.Series) -> float:
    """Most negative peak-to-trough decline (as a negative fraction)."""
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def drawdown_series(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return equity
    peak = equity.cummax()
    return equity / peak - 1.0


def historical_var_cvar(
    returns: pd.Series,
    alpha: float = 0.95,
) -> RiskReport:
    """Historical VaR / CVaR on simple (or log) portfolio returns.

    VaR is the α-quantile *loss*: -Q_{1-α}(r). CVaR is the mean loss in the
    tail at or beyond that quantile. A 95% daily VaR of 0.02 means the 5th
    percentile of the simulated daily return distribution was -2%.
    """
    r = pd.Series(returns).dropna().astype(float)
    n = int(r.size)
    if n < 20:
        return RiskReport(var_95=float("nan"), cvar_95=float("nan"), var_95_ann_approx=float("nan"), n_obs=n, method="historical")
    q = float(np.quantile(r.to_numpy(), 1.0 - alpha))
    var = -q
    tail = r[r <= q]
    cvar = -float(tail.mean()) if len(tail) else var
    # sqrt(252) scaling is a common Gaussian approximation, not a theorem for
    # historical VaR; reported only as a rough annualized comparison aid.
    var_ann = var * np.sqrt(252.0)
    return RiskReport(var_95=var, cvar_95=cvar, var_95_ann_approx=float(var_ann), n_obs=n, method="historical")


def cagr(equity: pd.Series, periods_per_year: float = 252.0) -> float:
    if equity.size < 2:
        return 0.0
    total = float(equity.iloc[-1] / equity.iloc[0])
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


def ann_vol(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    r = pd.Series(returns).dropna()
    if r.size < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: float = 252.0) -> float:
    r = pd.Series(returns).dropna()
    if r.size < 2:
        return 0.0
    excess = r - rf / periods_per_year
    vol = float(excess.std(ddof=1))
    if vol < 1e-16:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / vol)


def total_return(equity: pd.Series) -> float:
    if equity.size < 2:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)
