"""Optional Streamlit dashboard. Reads saved CLI results; does not require a live server.

Run: streamlit run src/streamlit_app.py
The CLI (python -m src.app) writes the same plots and tables so the project
is usable without Streamlit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd


def _load_saved():
    tab = _ROOT / "results" / "tables"
    fig = _ROOT / "results" / "figures"
    metrics = pd.read_csv(tab / "metrics.csv") if (tab / "metrics.csv").exists() else None
    weights = pd.read_csv(tab / "latest_weights.csv") if (tab / "latest_weights.csv").exists() else None
    summary = (tab / "summary.txt").read_text() if (tab / "summary.txt").exists() else ""
    meta = json.loads((tab / "run_meta.json").read_text()) if (tab / "run_meta.json").exists() else {}
    return metrics, weights, summary, meta, fig


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Adaptive Portfolio Manager", layout="wide")
    st.title("Adaptive Portfolio Manager")
    st.caption("Research simulation on public prices or a TOY panel. Not live trading.")

    capital = st.sidebar.number_input("Starting capital", min_value=1000.0, value=100000.0, step=1000.0)
    risk = st.sidebar.selectbox("Risk tolerance", ["low", "medium", "high"], index=1)
    rebalance = st.sidebar.selectbox("Rebalance", ["monthly", "quarterly"], index=0)
    run_now = st.sidebar.checkbox("Re-run pipeline (slow if LSTM)", value=False)

    if run_now:
        from src.app import load_config, parse_args, run_pipeline

        cfg = load_config(_ROOT / "configs" / "config.yaml")
        args = parse_args(
            ["--capital", str(capital), "--risk", risk, "--rebalance", rebalance]
        )
        with st.spinner("Running walk-forward pipeline…"):
            run_pipeline(cfg, args)

    metrics, weights, summary, meta, fig = _load_saved()
    st.markdown(summary or "No results yet. Run `python -m src.app` first.")
    if meta:
        st.write(
            f"Label: **{meta.get('label')}**. Model: `{meta.get('forecast_model')}`. "
            f"As-of {meta.get('as_of')}."
        )
    col1, col2 = st.columns(2)
    pie = fig / "pie.png"
    eq = fig / "equity.png"
    dd = fig / "drawdown.png"
    if pie.exists():
        col1.image(str(pie), caption="Recommended weights")
    if weights is not None:
        col2.dataframe(weights, use_container_width=True)
    if eq.exists():
        st.image(str(eq), caption="Equity curves")
    if dd.exists():
        st.image(str(dd), caption="Drawdowns")
    if metrics is not None:
        st.subheader("Backtest metrics")
        st.dataframe(metrics, use_container_width=True)
    st.caption("Transaction costs (10 bps of L1 turnover) are included in the simulated path.")


if __name__ == "__main__":
    main()
