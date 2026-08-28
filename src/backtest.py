"""Walk-forward rebalancing backtest with transaction costs.

At each rebalance date the optimizer may use only information dated on or
before that close. Between rebalances, weights drift with asset returns.
Turnover cost is ``tc_bps / 1e4 * sum_i |w_new - w_drifted|`` on rebalance
days (full one-way L1 of the trade, charged on the day's portfolio return).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.forecast import ForecastBundle
from src.optimizer import (
    Allocation,
    annualize_mu_vol,
    cov_from_corr_and_vols,
    equal_weight,
    hierarchical_risk_parity,
    ledoit_wolf_cov,
    mean_variance,
    sample_cov,
    sixty_forty,
)
from src.regime import apply_regime_filter
from src.risk import (
    RiskReport,
    ann_vol,
    cagr,
    drawdown_series,
    historical_var_cvar,
    max_drawdown,
    sharpe,
    total_return,
)


def rebalance_dates(
    index: pd.DatetimeIndex,
    frequency: str = "monthly",
    start: str | None = None,
) -> pd.DatetimeIndex:
    """Month-end (or quarter-end) dates that exist in the price calendar."""
    s = pd.Series(1, index=index)
    freq = "ME" if frequency.lower().startswith("month") else "QE"
    ends = s.resample(freq).last().index
    # map to last available session on or before each period end
    mapped = []
    idx = index.sort_values()
    for e in ends:
        pos = idx.searchsorted(e, side="right") - 1
        if pos >= 0:
            mapped.append(idx[pos])
    dates = pd.DatetimeIndex(sorted(set(mapped)))
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    dates = dates[dates <= idx.max()]
    return dates


@dataclass
class StrategyResult:
    name: str
    equity: pd.Series
    returns: pd.Series
    weights_history: pd.DataFrame
    turnover: pd.Series
    metrics: dict
    risk: RiskReport
    notes: list[str] = field(default_factory=list)


def _metrics_dict(
    equity: pd.Series,
    rets: pd.Series,
    turnover: pd.Series,
    rf: float,
    label: str,
) -> dict:
    risk = historical_var_cvar(rets, alpha=0.95)
    return {
        "label": label,
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "vol": ann_vol(rets),
        "sharpe": sharpe(rets, rf=rf),
        "max_dd": max_drawdown(equity),
        "avg_turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "n_rebalances": int((turnover > 0).sum()),
        "var_95_daily": risk.var_95,
        "cvar_95_daily": risk.cvar_95,
        "n_days": int(rets.dropna().size),
    }, risk


def _drift(prev_w: pd.Series, day_ret: pd.Series) -> pd.Series:
    gross = prev_w * (1.0 + day_ret.reindex(prev_w.index).fillna(0.0))
    s = float(gross.sum())
    if s <= 0:
        return prev_w * 0.0
    return gross / s


def _simple_from_log(log_r: pd.Series) -> pd.Series:
    return np.expm1(log_r)


def run_weight_path(
    name: str,
    weight_fn,
    prices: pd.DataFrame,
    log_returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    tc_bps: float = 10.0,
    initial_capital: float = 100_000.0,
    rf: float = 0.0,
    label: str = "FULL-public",
    buy_and_hold: bool = False,
) -> StrategyResult:
    """Generic engine. ``weight_fn(dt, drifted) -> (weights Series, cash, notes).``"""
    assets = list(log_returns.columns)
    simple = _simple_from_log(log_returns)
    # daily loop from first rebalance (inclusive) to last price date
    if len(dates) == 0:
        empty = pd.Series(dtype=float)
        dummy_risk = historical_var_cvar(empty)
        return StrategyResult(name, empty, empty, pd.DataFrame(), empty, {}, dummy_risk, ["no rebalance dates"])

    start = dates[0]
    px_idx = prices.loc[start:].index
    nav = initial_capital
    equity_pts = {}
    ret_pts = {}
    turn_pts = {}
    w_rows = []
    notes_all: list[str] = []
    prev_w = pd.Series(0.0, index=assets)
    cash = 1.0
    rebal_set = set(pd.Timestamp(d) for d in dates)
    first = True

    for dt in px_idx:
        r = simple.loc[dt] if dt in simple.index else pd.Series(0.0, index=assets)
        # portfolio return from previous close to this close using previous weights
        if first:
            port_r = 0.0
        else:
            asset_r = float((prev_w * r.reindex(prev_w.index).fillna(0.0)).sum())
            port_r = asset_r  # cash earns 0
        cost = 0.0
        if dt in rebal_set and (first or not buy_and_hold):
            drifted = prev_w if first else _drift(prev_w, r)
            new_w, cash, nts = weight_fn(dt, drifted)
            new_w = new_w.reindex(assets).fillna(0.0).clip(lower=0.0)
            if new_w.sum() > 1:
                new_w = new_w / new_w.sum()
            cash = float(max(0.0, 1.0 - new_w.sum()))
            l1 = float((new_w - drifted).abs().sum()) if not first else float(new_w.abs().sum())
            cost = (tc_bps / 1e4) * l1
            prev_w = new_w
            turn_pts[dt] = l1
            notes_all.extend(nts)
        elif not first:
            prev_w = _drift(prev_w, r)
            turn_pts[dt] = 0.0
        else:
            turn_pts[dt] = 0.0

        if not first:
            port_r = port_r - cost
            nav = nav * (1.0 + port_r)
            ret_pts[dt] = port_r
        else:
            # initial turnover is charged against starting capital
            nav = nav * (1.0 - cost)
            ret_pts[dt] = -cost
        equity_pts[dt] = nav
        row = prev_w.to_dict()
        row["cash"] = cash
        row["date"] = dt
        w_rows.append(row)
        first = False

    equity = pd.Series(equity_pts, dtype=float)
    equity.index = pd.to_datetime(equity.index)
    rets = pd.Series(ret_pts, dtype=float)
    rets.index = pd.to_datetime(rets.index)
    turnover = pd.Series(turn_pts, dtype=float)
    turnover.index = pd.to_datetime(turnover.index)
    # drop the zero first-day return from performance stats
    rets_perf = rets.iloc[1:] if len(rets) > 1 else rets
    metrics, risk = _metrics_dict(equity, rets_perf, turnover.loc[list(rebal_set)].sort_index(), rf, label)
    w_hist = pd.DataFrame(w_rows).set_index("date")
    return StrategyResult(name, equity, rets, w_hist, turnover, metrics, risk, notes_all)


def _trailing_slice(df: pd.DataFrame, dt, lookback: int) -> pd.DataFrame:
    hist = df.loc[:dt]
    if lookback and len(hist) > lookback:
        return hist.iloc[-lookback:]
    return hist


def build_all_strategies(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    forecasts: ForecastBundle,
    dates: pd.DatetimeIndex,
    *,
    max_weight: float = 0.25,
    min_weight: float = 0.0,
    risk_aversion: float = 3.0,
    target_vol: float | None = 0.12,
    rf: float = 0.0,
    cov_method: str = "ledoit_wolf",
    cov_lookback: int = 252,
    horizon: int = 21,
    tc_bps: float = 10.0,
    initial_capital: float = 100_000.0,
    label: str = "FULL-public",
    regime_cfg: dict | None = None,
    apply_regime_to: tuple[str, ...] = ("lstm_mv", "lstm_hrp"),
) -> dict[str, StrategyResult]:
    """Run ML adaptive strategies plus equal-weight, 60/40, and static MV."""
    names = list(returns.columns)
    regime_cfg = regime_cfg or {"enabled": False}
    mu_f = forecasts.mu_period
    vol_f = forecasts.vol_daily

    def cov_at(dt, vol_daily: pd.Series | None) -> pd.DataFrame:
        trail = _trailing_slice(returns, dt, cov_lookback)
        if vol_daily is not None:
            return cov_from_corr_and_vols(trail, vol_daily)
        if cov_method == "sample":
            return sample_cov(trail)
        return ledoit_wolf_cov(trail)

    def maybe_regime(w: pd.Series, dt) -> tuple[pd.Series, float, list[str]]:
        nw, cash, st = apply_regime_filter(
            w,
            prices,
            dt,
            enabled=bool(regime_cfg.get("enabled", True)),
            index=regime_cfg.get("index", "SPY"),
            fast=int(regime_cfg.get("fast_ma", 50)),
            slow=int(regime_cfg.get("slow_ma", 200)),
            risk_scale=float(regime_cfg.get("risk_scale", 0.5)),
            equity_tickers=regime_cfg.get("equity_tickers"),
            crypto_tickers=regime_cfg.get("crypto_tickers"),
            defensive_tickers=regime_cfg.get("defensive_tickers"),
            max_weight=max_weight,
        )
        return nw, cash, [st.note]

    def weights_lstm_mv(dt, drifted):
        if dt not in mu_f.index:
            loc = mu_f.index[mu_f.index <= dt]
            if len(loc) == 0:
                alloc = equal_weight(names, max_weight=max_weight)
                w, cash, nts = alloc.weights, alloc.cash, ["no forecast; equal weight"]
                w, cash, nts2 = maybe_regime(w, dt)
                return w, cash, nts + nts2
            row_dt = loc[-1]
        else:
            row_dt = dt
        mu_period = mu_f.loc[row_dt, names].astype(float)
        vol_d = vol_f.loc[row_dt, names].astype(float)
        mu_ann, vol_ann = annualize_mu_vol(mu_period, vol_d, horizon)
        cov = cov_at(dt, vol_d)
        alloc = mean_variance(
            mu_ann,
            cov,
            max_weight=max_weight,
            min_weight=min_weight,
            risk_aversion=risk_aversion,
            target_vol=target_vol,
            rf=rf,
        )
        w, cash, nts = alloc.weights, alloc.cash, list(alloc.notes)
        w, cash, nts2 = maybe_regime(w, dt)
        return w, cash, nts + nts2

    def weights_lstm_hrp(dt, drifted):
        loc = mu_f.index[mu_f.index <= dt]
        if len(loc) == 0:
            alloc = equal_weight(names, max_weight=max_weight)
            w, cash, nts = alloc.weights, alloc.cash, ["no forecast; equal weight"]
            w, cash, nts2 = maybe_regime(w, dt)
            return w, cash, nts + nts2
        row_dt = loc[-1]
        mu_period = mu_f.loc[row_dt, names].astype(float)
        vol_d = vol_f.loc[row_dt, names].astype(float)
        mu_ann, _ = annualize_mu_vol(mu_period, vol_d, horizon)
        cov = cov_at(dt, vol_d)
        alloc = hierarchical_risk_parity(
            cov, max_weight=max_weight, min_weight=min_weight, mu_ann=mu_ann, rf=rf
        )
        w, cash, nts = alloc.weights, alloc.cash, list(alloc.notes)
        w, cash, nts2 = maybe_regime(w, dt)
        return w, cash, nts + nts2

    def weights_ew(dt, drifted):
        alloc = equal_weight(names, max_weight=max_weight, cov_daily=cov_at(dt, None))
        return alloc.weights, alloc.cash, []

    def weights_static_mv(dt, drifted):
        trail = _trailing_slice(returns, dt, cov_lookback)
        mu_daily = trail.mean()
        mu_ann = mu_daily * 252.0
        cov = ledoit_wolf_cov(trail) if cov_method != "sample" else sample_cov(trail)
        alloc = mean_variance(
            mu_ann,
            cov,
            max_weight=max_weight,
            min_weight=min_weight,
            risk_aversion=risk_aversion,
            target_vol=target_vol,
            rf=rf,
        )
        return alloc.weights, alloc.cash, list(alloc.notes)

    def weights_6040(dt, drifted):
        bond = "TLT" if "TLT" in names else ("TLT_SYN" if "TLT_SYN" in names else "GLD")
        alloc = sixty_forty(names, equity_name="SPY", bond_name=bond)
        return alloc.weights, alloc.cash, list(alloc.notes)

    out: dict[str, StrategyResult] = {}
    out["lstm_mv"] = run_weight_path(
        "LSTM+MV", weights_lstm_mv, prices, returns, dates,
        tc_bps=tc_bps, initial_capital=initial_capital, rf=rf, label=label,
    )
    out["lstm_hrp"] = run_weight_path(
        "LSTM+HRP", weights_lstm_hrp, prices, returns, dates,
        tc_bps=tc_bps, initial_capital=initial_capital, rf=rf, label=label,
    )
    out["equal"] = run_weight_path(
        "Equal-weight", weights_ew, prices, returns, dates,
        tc_bps=tc_bps, initial_capital=initial_capital, rf=rf, label=label,
    )
    out["static_mv"] = run_weight_path(
        "Static MV (hist. mean)", weights_static_mv, prices, returns, dates,
        tc_bps=tc_bps, initial_capital=initial_capital, rf=rf, label=label,
    )
    out["bh_60_40"] = run_weight_path(
        "Buy-and-hold 60/40", weights_6040, prices, returns, dates,
        tc_bps=tc_bps, initial_capital=initial_capital, rf=rf, label=label,
        buy_and_hold=True,
    )
    return out


def metrics_table(results: dict[str, StrategyResult]) -> pd.DataFrame:
    rows = []
    for key, res in results.items():
        row = {"strategy": res.name, "key": key}
        row.update(res.metrics)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("strategy")
    return df


def latest_allocation_record(
    alloc: Allocation,
    cash: float,
    as_of,
    extra: dict | None = None,
) -> pd.DataFrame:
    w = alloc.weights.copy()
    df = w.rename("weight").to_frame()
    df["asset"] = df.index
    df["cash"] = cash
    df["as_of"] = pd.Timestamp(as_of)
    df["method"] = alloc.method
    df["expected_return_ann"] = alloc.expected_return
    df["expected_vol_ann"] = alloc.expected_vol
    df["sharpe"] = alloc.sharpe
    if extra:
        for k, v in extra.items():
            df[k] = v
    return df.reset_index(drop=True)
