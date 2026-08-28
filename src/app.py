"""CLI entry point: download, forecast, backtest, write figures/tables, print summary.

Usage
-----
python -m src.app --config configs/config.yaml
python src/app.py --capital 100000 --risk medium --rebalance monthly
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# allow `python src/app.py` from project root or src/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.backtest import (  # noqa: E402
    build_all_strategies,
    metrics_table,
    rebalance_dates,
)
from src.data_loader import load_panel  # noqa: E402
from src.forecast import walk_forward_forecasts  # noqa: E402
from src.optimizer import (  # noqa: E402
    annualize_mu_vol,
    cov_from_corr_and_vols,
    hierarchical_risk_parity,
    ledoit_wolf_cov,
    mean_variance,
)
from src.regime import apply_regime_filter  # noqa: E402
from src.risk import drawdown_series  # noqa: E402


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def risk_to_target_vol(cfg: dict, risk: str) -> float:
    mapping = cfg.get("portfolio", {}).get("target_vol", {})
    return float(mapping.get(risk, {"low": 0.08, "medium": 0.12, "high": 0.18}[risk]))


def _save_equity_figure(results: dict, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for res in results.values():
        if res.equity.empty:
            continue
        nav = res.equity / res.equity.iloc[0]
        ax.plot(nav.index, nav.values, label=res.name, linewidth=1.4)
    ax.set_ylabel("Growth of $1")
    ax.set_title(f"Walk-forward equity curves ({label})")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_drawdown_figure(results: dict, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for res in results.values():
        if res.equity.empty:
            continue
        dd = drawdown_series(res.equity)
        ax.plot(dd.index, dd.values, label=res.name, linewidth=1.2)
    ax.set_ylabel("Drawdown")
    ax.set_title(f"Drawdowns ({label})")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_pie(weights: pd.Series, cash: float, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = weights[weights > 1e-4].copy()
    if cash > 1e-4:
        w["CASH"] = cash
    if w.empty:
        w = pd.Series({"CASH": 1.0})
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.pie(w.values, labels=w.index, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8})
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plain_summary(
    alloc_w: pd.Series,
    cash: float,
    target_vol: float,
    risk: str,
    metrics: pd.DataFrame,
    label: str,
    tickers: list[str],
    failed: list[str],
    forecast_model: str,
) -> str:
    top = alloc_w.sort_values(ascending=False).head(5)
    top_s = ", ".join(f"{k} {v:.1%}" for k, v in top.items())
    lines = [
        f"Label: {label}. Forecast model: {forecast_model}. Universe ({len(tickers)}): {', '.join(tickers)}.",
    ]
    if failed:
        lines.append(f"Dropped tickers: {', '.join(failed)}.")
    lines.append(
        f"This allocation targets about {target_vol:.0%} annualized volatility "
        f"(risk tolerance: {risk}) under a long-only mean-variance rule with a "
        f"{0.25:.0%} per-name cap. Cash residual is {cash:.1%}."
    )
    lines.append(f"Largest holdings: {top_s}.")
    if "LSTM+MV" in metrics.index and "Equal-weight" in metrics.index:
        m = metrics.loc["LSTM+MV"]
        e = metrics.loc["Equal-weight"]
        lines.append(
            f"Backtest (transaction costs included): LSTM+MV CAGR {m['cagr']:.1%}, "
            f"vol {m['vol']:.1%}, Sharpe {m['sharpe']:.2f}, max DD {m['max_dd']:.1%}; "
            f"equal-weight CAGR {e['cagr']:.1%}, Sharpe {e['sharpe']:.2f}. "
            "These are historical simulations, not live trading results."
        )
    return " ".join(lines)


def run_pipeline(cfg: dict, args: argparse.Namespace) -> dict:
    root = _ROOT
    tickers = args.tickers.split(",") if args.tickers else list(cfg["universe"]["tickers"])
    tickers = [t.strip() for t in tickers if t.strip()]
    start = cfg["universe"].get("start", "2021-01-01")
    end = cfg["universe"].get("end")
    risk = args.risk
    target_vol = risk_to_target_vol(cfg, risk)
    freq = args.rebalance or cfg["rebalance"]["frequency"]
    capital = float(args.capital)
    fc_cfg = cfg["forecast"]
    if args.model:
        fc_cfg = dict(fc_cfg)
        fc_cfg["model"] = args.model

    print("[1/5] Loading price panel …")
    panel = load_panel(
        tickers=tickers,
        start=start,
        end=end,
        cache_dir=root / cfg["data"]["cache_dir"],
        processed_dir=root / cfg["data"]["processed_dir"],
        max_ffill_days=int(cfg["data"].get("max_ffill_days", 5)),
        force_toy=bool(args.toy),
    )
    print(f"    {panel.label}  n={len(panel.tickers)}  {panel.prices.index.min().date()} → {panel.prices.index.max().date()}")
    for n in panel.notes:
        print(f"    note: {n}")

    bt_start = cfg["backtest"].get("start", "2023-01-01")
    dates = rebalance_dates(panel.prices.index, frequency=freq, start=bt_start)
    print(f"[2/5] {len(dates)} {freq} rebalance dates from {dates.min().date() if len(dates) else 'n/a'}")

    models_dir = root / cfg.get("output", {}).get("models_dir", "models")
    print(f"[3/5] Walk-forward forecasts ({fc_cfg.get('model', 'lstm')}, seed={fc_cfg.get('seed', 42)}) …")
    forecasts = walk_forward_forecasts(
        panel.returns,
        dates,
        model=fc_cfg.get("model", "lstm"),
        lookback=int(fc_cfg.get("lookback", 20)),
        horizon=int(fc_cfg.get("horizon", 21)),
        hidden_size=int(fc_cfg.get("hidden_size", 16)),
        num_layers=int(fc_cfg.get("num_layers", 1)),
        epochs=int(fc_cfg.get("epochs", 8)),
        batch_size=int(fc_cfg.get("batch_size", 64)),
        learning_rate=float(fc_cfg.get("learning_rate", 0.01)),
        seed=int(fc_cfg.get("seed", 42)),
        retrain_every=int(fc_cfg.get("retrain_every", 4)),
        min_train_obs=int(fc_cfg.get("min_train_obs", 252)),
        models_dir=models_dir,
    )
    for n in forecasts.notes:
        print(f"    note: {n}")
    print(f"    model used: {forecasts.model_used}")

    print("[4/5] Backtests (10 bps per unit L1 turnover) …")
    results = build_all_strategies(
        panel.prices,
        panel.returns,
        forecasts,
        dates,
        max_weight=float(cfg["portfolio"]["max_weight"]),
        min_weight=float(cfg["portfolio"].get("min_weight", 0.0)),
        risk_aversion=float(cfg["portfolio"].get("risk_aversion", 3.0)),
        target_vol=target_vol,
        rf=float(cfg["portfolio"].get("rf", 0.0)),
        cov_method=cfg["portfolio"].get("cov_method", "ledoit_wolf"),
        cov_lookback=int(cfg["portfolio"].get("cov_lookback", 252)),
        horizon=int(fc_cfg.get("horizon", 21)),
        tc_bps=float(cfg["rebalance"].get("transaction_cost_bps", 10)),
        initial_capital=capital,
        label=panel.label,
        regime_cfg=cfg.get("regime", {}),
    )
    table = metrics_table(results)
    fig_dir = root / cfg["output"]["figures_dir"]
    tab_dir = root / cfg["output"]["tables_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(tab_dir / "metrics.csv")
    _save_equity_figure(results, fig_dir / "equity.png", panel.label)
    _save_drawdown_figure(results, fig_dir / "drawdown.png", panel.label)

    # latest recommended allocation from last forecast row
    last_dt = forecasts.mu_period.index[-1]
    names = list(panel.returns.columns)
    mu_period = forecasts.mu_period.loc[last_dt, names].astype(float)
    vol_d = forecasts.vol_daily.loc[last_dt, names].astype(float)
    mu_ann, _ = annualize_mu_vol(mu_period, vol_d, int(fc_cfg.get("horizon", 21)))
    trail = panel.returns.loc[:last_dt].iloc[-int(cfg["portfolio"].get("cov_lookback", 252)) :]
    cov = cov_from_corr_and_vols(trail, vol_d)
    alloc = mean_variance(
        mu_ann,
        cov,
        max_weight=float(cfg["portfolio"]["max_weight"]),
        min_weight=float(cfg["portfolio"].get("min_weight", 0.0)),
        risk_aversion=float(cfg["portfolio"].get("risk_aversion", 3.0)),
        target_vol=target_vol,
        rf=float(cfg["portfolio"].get("rf", 0.0)),
    )
    w, cash, st = apply_regime_filter(
        alloc.weights,
        panel.prices,
        last_dt,
        enabled=bool(cfg.get("regime", {}).get("enabled", True)),
        index=cfg.get("regime", {}).get("index", "SPY"),
        fast=int(cfg.get("regime", {}).get("fast_ma", 50)),
        slow=int(cfg.get("regime", {}).get("slow_ma", 200)),
        risk_scale=float(cfg.get("regime", {}).get("risk_scale", 0.5)),
        equity_tickers=cfg.get("regime", {}).get("equity_tickers"),
        crypto_tickers=cfg.get("regime", {}).get("crypto_tickers"),
        defensive_tickers=cfg.get("regime", {}).get("defensive_tickers"),
        max_weight=float(cfg["portfolio"]["max_weight"]),
    )
    hrp = hierarchical_risk_parity(
        cov,
        max_weight=float(cfg["portfolio"]["max_weight"]),
        mu_ann=mu_ann,
        rf=float(cfg["portfolio"].get("rf", 0.0)),
    )

    weights_df = pd.DataFrame(
        {
            "asset": w.index,
            "weight_mv_regime": w.values,
            "weight_hrp": hrp.weights.reindex(w.index).fillna(0.0).values,
            "forecast_mu_period": mu_period.reindex(w.index).values,
            "forecast_vol_daily": vol_d.reindex(w.index).values,
            "forecast_mu_ann": mu_ann.reindex(w.index).values,
        }
    )
    weights_df["cash_mv_regime"] = cash
    weights_df["as_of"] = last_dt
    weights_df["label"] = panel.label
    weights_df["target_vol"] = target_vol
    weights_df["regime_note"] = st.note
    weights_df.to_csv(tab_dir / "latest_weights.csv", index=False)

    # also a one-row-per-strategy weights snapshot
    last_w_rows = []
    for key, res in results.items():
        if res.weights_history.empty:
            continue
        last = res.weights_history.iloc[-1]
        rec = last.to_dict()
        rec["strategy"] = res.name
        rec["key"] = key
        last_w_rows.append(rec)
    if last_w_rows:
        pd.DataFrame(last_w_rows).to_csv(tab_dir / "last_backtest_weights.csv", index=False)

    _save_pie(
        w,
        cash,
        fig_dir / "pie.png",
        f"Recommended MV+regime weights as of {last_dt.date()} ({panel.label})",
    )

    # forecast table
    ftab = pd.concat(
        {
            "mu_period": forecasts.mu_period.iloc[-1],
            "vol_daily": forecasts.vol_daily.iloc[-1],
        },
        axis=1,
    )
    ftab.to_csv(tab_dir / "latest_forecasts.csv")

    summary = _plain_summary(
        w, cash, target_vol, risk, table, panel.label, panel.tickers, panel.failed_tickers, forecasts.model_used
    )
    (tab_dir / "summary.txt").write_text(summary + "\n")
    meta = {
        "label": panel.label,
        "source": panel.source,
        "tickers": panel.tickers,
        "failed_tickers": panel.failed_tickers,
        "notes": panel.notes + forecasts.notes,
        "forecast_model": forecasts.model_used,
        "n_rebalances": int(len(dates)),
        "as_of": str(last_dt.date()),
        "target_vol": target_vol,
        "risk": risk,
        "regime": st.note,
        "summary": summary,
    }
    (tab_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    print("[5/5] Wrote figures and tables.")
    print()
    print("=" * 72)
    print("LATEST RECOMMENDED WEIGHTS (MV + regime filter)")
    print("=" * 72)
    show = w[w > 1e-4].sort_values(ascending=False)
    for k, v in show.items():
        print(f"  {k:10s}  {v:7.2%}")
    print(f"  {'CASH':10s}  {cash:7.2%}")
    print(f"  as-of {last_dt.date()}  target vol {target_vol:.0%}  {panel.label}")
    print(f"  {st.note}")
    print()
    print("BACKTEST COMPARISON")
    cols = ["label", "total_return", "cagr", "vol", "sharpe", "max_dd", "avg_turnover", "var_95_daily", "cvar_95_daily"]
    cols = [c for c in cols if c in table.columns]
    with pd.option_context("display.float_format", "{:.4f}".format, "display.max_columns", 20, "display.width", 120):
        print(table[cols].to_string())
    print()
    print(summary)
    print()
    print(f"Figures: {fig_dir}")
    print(f"Tables:  {tab_dir}")
    return {
        "panel": panel,
        "forecasts": forecasts,
        "results": results,
        "table": table,
        "weights": w,
        "cash": cash,
        "summary": summary,
        "meta": meta,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptive Portfolio Manager (research backtest, not live trading).")
    p.add_argument("--config", default=str(_ROOT / "configs" / "config.yaml"))
    p.add_argument("--capital", type=float, default=100_000.0, help="Starting capital for the simulated path.")
    p.add_argument("--risk", choices=["low", "medium", "high"], default="medium")
    p.add_argument("--rebalance", choices=["monthly", "quarterly"], default=None)
    p.add_argument("--tickers", default=None, help="Comma-separated list; default from config.yaml")
    p.add_argument("--model", choices=["lstm", "ridge", "transformer"], default=None)
    p.add_argument("--toy", action="store_true", help="Skip Yahoo and use the labeled TOY panel.")
    p.add_argument("--skip-train-if-models", action="store_true", help="Ignored placeholder for Makefile.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    # CLI capital overrides
    cfg.setdefault("backtest", {})["initial_capital"] = args.capital
    run_pipeline(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
