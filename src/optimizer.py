"""Long-only portfolio constructors: mean-variance and hierarchical risk parity.

Implemented in numpy/scipy. PyPortfolioOpt is optional and unused. HRP follows
the correlation-distance / linkage / recursive-bisection procedure described
by López de Prado (2016, 2018). This is an independent implementation, not a
copy of any GPL package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf


@dataclass
class Allocation:
    weights: pd.Series
    cash: float
    expected_return: float  # annualized
    expected_vol: float  # annualized
    sharpe: float
    method: str
    notes: list[str]


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit–Wolf shrinkage covariance of daily log returns."""
    x = returns.dropna(how="any").to_numpy(dtype=np.float64)
    cols = list(returns.columns)
    if x.shape[0] < max(10, x.shape[1] + 2):
        sample = np.cov(returns.fillna(0.0).to_numpy().T, ddof=1)
        # simple constant-correlation-style ridge if LW is infeasible
        sample = np.atleast_2d(sample)
        sample = 0.9 * sample + 0.1 * np.diag(np.diag(sample) + 1e-8)
        return pd.DataFrame(sample, index=cols, columns=cols)
    lw = LedoitWolf().fit(x)
    return pd.DataFrame(lw.covariance_, index=cols, columns=cols)


def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.dropna(how="any").cov()


def annualize_mu_vol(mu_period: pd.Series, vol_daily: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series]:
    """Convert horizon-sum log-return forecasts and daily vol to annualized."""
    steps = 252.0 / max(horizon, 1)
    mu_ann = mu_period * steps
    vol_ann = vol_daily * np.sqrt(252.0)
    return mu_ann, vol_ann


def cov_from_corr_and_vols(
    trailing_returns: pd.DataFrame,
    vol_daily: pd.Series,
) -> pd.DataFrame:
    """Historical correlation, forecasted vols on the diagonal (documented).

    Σ_ij = ρ_ij * σ_i * σ_j with σ from the forecast and ρ from trailing
    daily log returns (Ledoit–Wolf correlation via shrunk covariance).
    """
    cov_hist = ledoit_wolf_cov(trailing_returns)
    d = np.sqrt(np.clip(np.diag(cov_hist.to_numpy()), 1e-12, None))
    corr = cov_hist.to_numpy() / np.outer(d, d)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    sig = vol_daily.reindex(cov_hist.index).fillna(vol_daily.median()).to_numpy()
    sig = np.clip(sig, 1e-8, None)
    cov = corr * np.outer(sig, sig)
    return pd.DataFrame(cov, index=cov_hist.index, columns=cov_hist.columns)


def apply_box_constraints(
    weights: np.ndarray,
    max_weight: float = 0.25,
    min_weight: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Long-only, cap at max_weight, leftover is cash (does not lever)."""
    w = np.asarray(weights, dtype=np.float64).copy()
    w = np.nan_to_num(w, nan=0.0)
    w = np.maximum(w, 0.0)
    if w.sum() > 1.0:
        w = w / w.sum()
    w = np.minimum(w, max_weight)
    if min_weight > 0:
        w = np.maximum(w, min_weight)
        if w.sum() > 1.0:
            # reduce names above min_weight
            excess = w.sum() - 1.0
            room = w - min_weight
            room = np.maximum(room, 0.0)
            if room.sum() > 0:
                w = w - excess * room / room.sum()
            w = np.maximum(w, 0.0)
    w = np.minimum(w, max_weight)
    w = np.maximum(w, 0.0)
    # if still over-invested, scale down (cash absorbs)
    if w.sum() > 1.0:
        w = w / w.sum()
        w = np.minimum(w, max_weight)
    cash = float(max(0.0, 1.0 - w.sum()))
    # numerical dust
    if cash < 1e-10:
        cash = 0.0
        if w.sum() > 0:
            w = w / w.sum()
            w = np.minimum(w, max_weight)
            cash = float(max(0.0, 1.0 - w.sum()))
    return w, cash


def _portfolio_stats(w: np.ndarray, mu_ann: np.ndarray, cov_daily: np.ndarray, rf: float) -> tuple[float, float, float]:
    mu = float(w @ mu_ann)
    vol = float(np.sqrt(max(w @ cov_daily @ w, 0.0) * 252.0))
    sharpe = (mu - rf) / vol if vol > 1e-12 else 0.0
    return mu, vol, sharpe


def mean_variance(
    mu_ann: pd.Series,
    cov_daily: pd.DataFrame,
    *,
    max_weight: float = 0.25,
    min_weight: float = 0.0,
    risk_aversion: float = 3.0,
    target_vol: float | None = None,
    rf: float = 0.0,
) -> Allocation:
    """Long-only Markowitz utility max w'μ − (λ/2) w'Σ_ann w, box constraints.

    ``cov_daily`` is the daily covariance; it is scaled by 252 in the objective.
    If ``target_vol`` is set and the solution vol exceeds it, weights are
    scaled toward cash (no leverage).
    """
    names = list(mu_ann.index)
    mu = mu_ann.reindex(names).fillna(0.0).to_numpy()
    cov = cov_daily.reindex(index=names, columns=names).fillna(0.0).to_numpy()
    cov = 0.5 * (cov + cov.T)
    n = len(names)
    if n == 0:
        return Allocation(pd.Series(dtype=float), 1.0, 0.0, 0.0, 0.0, "mv", ["empty"])

    cov_ann = cov * 252.0
    lam = max(float(risk_aversion), 1e-6)

    def objective(w: np.ndarray) -> float:
        return float(-(w @ mu) + 0.5 * lam * (w @ cov_ann @ w))

    w0 = np.full(n, 1.0 / n)
    bounds = [(min_weight, max_weight)] * n
    cons = [{"type": "ineq", "fun": lambda w: 1.0 - np.sum(w)}]  # sum <= 1 (cash ok)
    # if fully invested is feasible, also try equality via a nudge: prefer investing
    # by adding a tiny cash penalty already in the objective (rf=0 means cash has 0 return)

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 300, "ftol": 1e-9, "disp": False},
    )
    w = res.x if res.success else w0
    w, cash = apply_box_constraints(w, max_weight=max_weight, min_weight=min_weight)
    notes = []
    if not res.success:
        notes.append(f"SLSQP did not report success ({res.message}); using projected weights.")

    mu_p, vol_p, sh = _portfolio_stats(w, mu, cov, rf)
    if target_vol is not None and vol_p > target_vol + 1e-12 and vol_p > 0:
        scale = target_vol / vol_p
        w = w * scale
        w, cash = apply_box_constraints(w, max_weight=max_weight, min_weight=0.0)
        mu_p, vol_p, sh = _portfolio_stats(w, mu, cov, rf)
        notes.append(f"Scaled toward cash to target vol {target_vol:.2%}.")

    return Allocation(
        weights=pd.Series(w, index=names),
        cash=cash,
        expected_return=mu_p,
        expected_vol=vol_p,
        sharpe=sh,
        method="mv",
        notes=notes,
    )


def _correl_distance(corr: np.ndarray) -> np.ndarray:
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    return dist


def _quasi_diag(link: np.ndarray) -> list[int]:
    """Seriation: sort leaves so that clusters sit on the diagonal."""
    link = np.asarray(link, dtype=float)
    n_items = int(link[-1, 3])
    order = [int(link[-1, 0]), int(link[-1, 1])]
    while max(order) >= n_items:
        new_order: list[int] = []
        for idx in order:
            if idx < n_items:
                new_order.append(idx)
            else:
                row = int(idx - n_items)
                new_order.append(int(link[row, 0]))
                new_order.append(int(link[row, 1]))
        order = new_order
    return order


def _cluster_var(cov: np.ndarray, items: list[int]) -> float:
    sub = cov[np.ix_(items, items)]
    diag = np.clip(np.diag(sub), 1e-12, None)
    ivp = 1.0 / diag
    ivp = ivp / ivp.sum()
    return float(ivp @ sub @ ivp)


def _recursive_bisection(cov: np.ndarray, sort_ix: list[int]) -> np.ndarray:
    w = np.ones(len(sort_ix), dtype=np.float64)
    idx_map = {orig: k for k, orig in enumerate(sort_ix)}
    clusters = [list(sort_ix)]
    while clusters:
        new_clusters: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]
            if not left or not right:
                continue
            var_l = _cluster_var(cov, left)
            var_r = _cluster_var(cov, right)
            alpha = 1.0 - var_l / (var_l + var_r + 1e-16)
            for i in left:
                w[idx_map[i]] *= alpha
            for i in right:
                w[idx_map[i]] *= 1.0 - alpha
            if len(left) > 1:
                new_clusters.append(left)
            if len(right) > 1:
                new_clusters.append(right)
        clusters = new_clusters
    # w is in sort_ix order; scatter back to original index order
    out = np.zeros(cov.shape[0], dtype=np.float64)
    for pos, orig in enumerate(sort_ix):
        out[orig] = w[pos]
    s = out.sum()
    if s > 0:
        out = out / s
    return out


def hierarchical_risk_parity(
    cov_daily: pd.DataFrame,
    *,
    max_weight: float = 0.25,
    min_weight: float = 0.0,
    mu_ann: pd.Series | None = None,
    rf: float = 0.0,
) -> Allocation:
    """HRP: correlation distance, single linkage, recursive inverse-var split."""
    names = list(cov_daily.columns)
    cov = cov_daily.reindex(index=names, columns=names).to_numpy(dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    n = len(names)
    notes: list[str] = []
    if n == 1:
        w = np.array([min(1.0, max_weight)])
        cash = 1.0 - float(w[0])
        return Allocation(pd.Series(w, index=names), cash, 0.0, 0.0, 0.0, "hrp", notes)

    d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(d, d)
    dist = _correl_distance(corr)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    order = _quasi_diag(link)
    w = _recursive_bisection(cov, order)
    w, cash = apply_box_constraints(w, max_weight=max_weight, min_weight=min_weight)

    if mu_ann is None:
        mu_vec = np.zeros(n)
    else:
        mu_vec = mu_ann.reindex(names).fillna(0.0).to_numpy()
    mu_p, vol_p, sh = _portfolio_stats(w, mu_vec, cov, rf)
    return Allocation(
        weights=pd.Series(w, index=names),
        cash=cash,
        expected_return=mu_p,
        expected_vol=vol_p,
        sharpe=sh,
        method="hrp",
        notes=notes,
    )


def equal_weight(
    names: list[str],
    *,
    max_weight: float = 0.25,
    cov_daily: pd.DataFrame | None = None,
    mu_ann: pd.Series | None = None,
    rf: float = 0.0,
) -> Allocation:
    n = len(names)
    w = np.full(n, 1.0 / n) if n else np.array([])
    w, cash = apply_box_constraints(w, max_weight=max_weight, min_weight=0.0)
    mu_vec = mu_ann.reindex(names).fillna(0.0).to_numpy() if mu_ann is not None else np.zeros(n)
    cov = (
        cov_daily.reindex(index=names, columns=names).to_numpy()
        if cov_daily is not None
        else np.eye(n) * 1e-4
    )
    mu_p, vol_p, sh = _portfolio_stats(w, mu_vec, cov, rf) if n else (0.0, 0.0, 0.0)
    return Allocation(pd.Series(w, index=names), cash, mu_p, vol_p, sh, "equal", [])


def sixty_forty(
    names: list[str],
    equity_name: str = "SPY",
    bond_name: str = "TLT",
    *,
    cov_daily: pd.DataFrame | None = None,
    mu_ann: pd.Series | None = None,
    rf: float = 0.0,
) -> Allocation:
    """Buy-and-hold style 60/40 weights (used as the initial allocation)."""
    w = pd.Series(0.0, index=names)
    notes = []
    eq = equity_name if equity_name in names else None
    bd = bond_name if bond_name in names else None
    if bd is None:
        for cand in ("TLT_SYN", "GLD"):
            if cand in names:
                bd = cand
                notes.append(f"60/40 bond leg substituted with {bd}.")
                break
    if eq is None:
        for cand in ("QQQ", "IWM", "EFA"):
            if cand in names:
                eq = cand
                notes.append(f"60/40 equity leg substituted with {eq}.")
                break
    if eq is None or bd is None:
        # last resort: equal split of first two names
        if len(names) >= 2:
            eq, bd = names[0], names[1]
            notes.append(f"60/40 fallback to {eq}/{bd}.")
        elif len(names) == 1:
            w[names[0]] = 1.0
            notes.append("60/40 degenerate: single asset.")
            return Allocation(w, 0.0, 0.0, 0.0, 0.0, "60_40", notes)
    w[eq] = 0.60
    w[bd] = 0.40
    mu_vec = mu_ann.reindex(names).fillna(0.0).to_numpy() if mu_ann is not None else np.zeros(len(names))
    cov = (
        cov_daily.reindex(index=names, columns=names).to_numpy()
        if cov_daily is not None
        else np.eye(len(names)) * 1e-4
    )
    mu_p, vol_p, sh = _portfolio_stats(w.to_numpy(), mu_vec, cov, rf)
    return Allocation(w, 0.0, mu_p, vol_p, sh, "60_40", notes)
