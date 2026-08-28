# Adaptive Portfolio Manager

A research tool that **forecasts next-period return and volatility** for a small multi-asset universe, **constructs long-only portfolios** (mean-variance and hierarchical risk parity), and **rebalances on a walk-forward calendar** with transaction costs and a moving-average regime filter.

Copyright (c) 2026 Hossein Tabasi. MIT License.

This is **not** a live trading system. It does not place orders, does not claim live PnL, and does not treat a Yahoo-backed historical simulation as a performance track record.

---

## What it does

1. Downloads daily adjusted OHLCV from Yahoo Finance (`yfinance`) for a configurable ticker list (US stocks/ETFs plus a couple of cryptos). Failed names are dropped. If **every** download fails, a labeled **TOY** synthetic panel is generated so the code still runs.
2. Cleans the panel (weekday calendar, limited forward-fill, log returns) and caches it under `data/`.
3. Trains a **tiny LSTM** (1 layer, hidden 16, lookback 20, CPU) on **expanding windows** to predict the next ~21-day return **and** daily volatility. Features at date *t* never include *t+1*. A sklearn **Ridge** fallback is used if PyTorch is unavailable. A one-layer Transformer is implemented behind a config flag; the default is LSTM.
4. Builds two portfolios at each rebalance date:
   - **Mean-variance** (Markowitz) with forecasted means, a covariance that uses **Ledoit–Wolf correlations** and **forecasted vols**, long-only, 25% max weight, optional target volatility.
   - **Hierarchical risk parity** (López de Prado style): correlation distance, single linkage, recursive bisection. Implemented in numpy/scipy; not vendored from a GPL library.
5. **Monthly** (default) or quarterly rebalancing. **10 bps** of L1 turnover is charged. If the 50-day MA of SPY is below the 200-day MA (using only data on the rebalance date), equity and crypto weights are scaled toward TLT/GLD/cash.
6. Compares LSTM+MV and LSTM+HRP with equal-weight, buy-and-hold 60/40 (SPY/TLT), and static mean-variance on historical averages (no ML).
7. Writes equity/drawdown/pie figures, a metrics CSV, `latest_weights.csv`, and a short console summary.

---

## Setup

Python 3.11+. PyTorch is required only for LSTM/Transformer; tests and the Ridge path run without it.

```bash
# from this directory; the shared env at /workspace/.venv already has torch
/workspace/.venv/bin/pip install -r requirements.txt
# or
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

---

## How to run

```bash
# full pipeline (download, walk-forward LSTM, backtest, figures)
python -m src.app --config configs/config.yaml --capital 100000 --risk medium --rebalance monthly

# equivalent
python src/app.py --capital 100000 --risk medium

# Ridge-only (fast; no torch train)
python -m src.app --model ridge

# labeled synthetic panel (no Yahoo)
python -m src.app --toy

# optional dashboard (CLI already wrote the plots)
streamlit run src/streamlit_app.py
```

Risk tolerance maps to a **target annualized volatility** in `configs/config.yaml`: low 8%, medium 12%, high 18%. If the unconstrained long-only solution is louder than the target, weights are scaled toward cash (no leverage).

Unit tests (no heavy training):

```bash
python -m pytest -q -m "not slow"
# or
make test
```

---

## Method (short)

- **No look-ahead.** Supervised samples use a lookback window of daily log returns ending at *t* and a target window *(t+1, t+horizon]*. At as-of date *T*, training keeps only samples whose target window has already ended. Inference uses the last lookback window ending on *T*.
- **Covariance.** Historical Ledoit–Wolf correlation of trailing daily returns, combined with LSTM (or Ridge) volatility forecasts: Σ_ij = ρ_ij σ_i σ_j. Static MV uses historical means and Ledoit–Wolf covariance only.
- **HRP.** Distance √((1−ρ)/2), single-linkage clustering, quasi-diagonalization, recursive inverse-variance bisection, then the same long-only / max-weight / cash projection.
- **Costs.** On a rebalance close, turnover = Σ |w_new − w_drifted|; cost = 10 bps × turnover, subtracted from that day’s portfolio return. Initial allocation is also charged.
- **Regime.** SMA(50) < SMA(200) on SPY (or a substitute index if SPY is missing) → multiply equity/crypto weights by 0.5 and pour the residual into TLT/GLD, then cash, respecting the 25% cap.

---

## Outputs

| Path | Contents |
| --- | --- |
| `results/figures/equity.png` | Growth of $1 for all strategies |
| `results/figures/drawdown.png` | Drawdown paths |
| `results/figures/pie.png` | Latest recommended MV+regime weights |
| `results/tables/metrics.csv` | Total return, CAGR, vol, Sharpe, max DD, turnover, VaR/CVaR |
| `results/tables/latest_weights.csv` | Recommended allocation + forecasts |
| `results/tables/run_meta.json` | FULL-public vs TOY, tickers, notes |
| `models/` | Last LSTM `state_dict` + `metadata.json` |

Every table is tagged **FULL-public** (Yahoo) or **TOY**. If TLT is missing, a synthetic bond `TLT_SYN` is added and documented; that 60/40 leg is not a listed TLT series.

---

## Limitations (read these)

- Yahoo-adjusted closes are a convenient public sample, not a research-grade total-return database. Corporate actions, FX, borrow, and overnight gaps are not modeled beyond what `auto_adjust=True` provides.
- The LSTM is deliberately small (minutes on CPU, few epochs). It is a **baseline**, not a tuned return model. Walk-forward forecasts can be noisy; that is expected.
- Long-only, 25% caps, and a two-MA filter are design choices, not optimal policy.
- Sharpe ratios use rf = 0 unless you change the config. Historical 95% VaR/CVaR is a sample quantile of the **simulated** daily returns, not a forward risk forecast.
- Crypto trades on weekends; the panel is aligned to **weekdays** so weekend crypto prints do not dilute equity volatility. That choice affects BTC/ETH returns on Mondays.
- **Do not** treat the backtest as evidence that the strategy will make money. It is a software demonstration with an honest cost and a leakage-aware forecast loop.

See `docs/REPORT.md` for a longer write-up and the numbers from the run that produced the files in `results/`.

---

## Citation

See `CITATION.cff`. Please cite this software if you use it, and cite López de Prado (2016/2018) for HRP and Ledoit & Wolf (2004) for covariance shrinkage.

---

## Project layout

```
adaptive-portfolio-manager/
  configs/config.yaml
  src/{data_loader,forecast,optimizer,backtest,regime,risk,app,streamlit_app}.py
  tests/test_no_lookahead.py
  tests/test_optimizer.py
  results/figures/  results/tables/
  docs/REPORT.md
```
