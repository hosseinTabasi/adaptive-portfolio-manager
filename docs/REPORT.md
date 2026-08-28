# Adaptive Portfolio Manager — method and run report

Hossein Tabasi, August 2026.

This note describes the software in this repository and the numbers produced by one walk-forward run on public Yahoo Finance prices. The run is labeled **FULL-public**. It is a historical simulation with transaction costs. It is not a live-trading record, and it is not an argument that the LSTM allocator will outperform equal weight out of sample.

## 1. Purpose

The practical question is narrow: if one replaces trailing sample means with a small neural (or linear) forecast of next-month return and volatility, and then feeds those forecasts into two standard long-only constructors (mean-variance and hierarchical risk parity), what does a leakage-aware monthly backtest look like against three simple benchmarks?

The design constraints were chosen so that a reader can reproduce the path on a laptop:

- long-only, 25 percent maximum weight, cash allowed;
- monthly rebalancing from 2023-01-31 through 2026-08-28;
- 10 basis points charged on L1 turnover;
- a two-moving-average regime overlay that uses only information known on the rebalance date;
- a one-layer LSTM with hidden size 16, lookback 20, eight epochs, seed 42, retrained every four rebalance dates.

Nothing in the pipeline estimates capacity, market impact beyond the flat 10 bp tax, or implementation shortfall.

## 2. Data

Default universe (configurable in `configs/config.yaml`): AAPL, MSFT, SPY, QQQ, GLD, TLT, IWM, EFA, BTC-USD, ETH-USD. The mix is eight listed US names (stocks and ETFs) plus two cryptocurrencies.

On this run, `yfinance` returned all ten series. Failed tickers: none. Synthetic bond: not used. Panel label: **FULL-public**. Date range after cleaning: 2021-01-04 to 2026-08-28, 1,475 weekday rows.

Cleaning:

1. Adjusted close from Yahoo (`auto_adjust=True`).
2. Reindex to a weekday calendar. Crypto weekend prints are dropped so that Saturday/Sunday BTC and ETH observations do not dilute equity volatility. Monday crypto returns therefore span the weekend.
3. Forward-fill at most five sessions (holidays). Rows that still contain any NaN are dropped. Leading NaNs are removed until every live column has an observation.
4. Log returns \(r_{i,t} = \log(P_{i,t}/P_{i,t-1})\).

Processed prices and returns are written to `data/processed/`. The raw per-ticker cache under `data/cache/` is gitignored.

Yahoo is a convenience source. It is not CRSP, not a total-return swap tape, and not a point-in-time fundamental database. On the last calendar day of this run (2026-08-28) several equity closes were unchanged versus 2026-08-27 while BTC and ETH moved, which is consistent with a same-day snapshot in which the cash session had not printed a new close. The backtest treats those repeats as zero equity returns for that session.

If every download had failed, `src/data_loader.py` would have built a labeled TOY factor-model panel and tagged every table TOY. That branch was not taken here.

## 3. Forecasting

For each asset the supervised problem is:

- features: the last \(L=20\) daily log returns ending at \(t\);
- target return: the sum of log returns on \((t+1,\ldots,t+H]\) with \(H=21\);
- target volatility: the sample standard deviation of those \(H\) daily returns.

The feature window and the target window do not overlap. At as-of date \(T\), a sample is eligible for training only if its target window has already ended (\(t+H \le T\)). Inference at a rebalance date uses the lookback window that ends on that date. Unit tests in `tests/test_no_lookahead.py` check the index arithmetic and a spike-after-as-of case on the Ridge path.

The default model is a one-layer LSTM mapping \(\mathbb{R}^{L \times 1}\) to two heads (period return and daily vol). Targets are standardized on the training fold only. Training is Adam, MSE on both heads, gradient clip 1.0, batch size 64, learning rate 0.01, eight epochs, CPU, seed 42. Models are refit every four monthly dates (expanding window). The last state dict per ticker is stored under `models/` together with `metadata.json`.

A one-layer Transformer (d_model 16, 2 heads) is implemented behind `forecast.model: transformer`. It was not the default for this run. If PyTorch is missing, or `APM_SKIP_TORCH=1`, the same walk-forward loop uses sklearn Ridge on the flattened lookback vector. That fallback is what CI exercises.

The LSTM is intentionally small. Eight epochs on a few hundred expanding-window sequences will not extract a stable risk premium. The forecasts should be read as a noisy, leakage-aware input to the optimizer, not as a claim that sequential models forecast monthly equity returns well.

Last forecast row (as-of 2026-08-28), period expected log return and daily vol:

| Asset | \(\hat\mu_{21d}\) | \(\hat\sigma_{daily}\) |
| --- | ---: | ---: |
| AAPL | 0.0131 | 0.0158 |
| MSFT | 0.0087 | 0.0152 |
| SPY | 0.0113 | 0.0095 |
| QQQ | 0.0125 | 0.0128 |
| GLD | 0.0153 | 0.0099 |
| TLT | −0.0061 | 0.0094 |
| IWM | 0.0056 | 0.0136 |
| EFA | 0.0076 | 0.0094 |
| BTC-USD | 0.0066 | 0.0329 |
| ETH-USD | 0.0020 | 0.0435 |

GLD had the highest predicted 21-day return and cryptos the highest predicted vol. TLT was the only negative mean forecast. These are model outputs on this date, not equilibrium expected returns.

## 4. Portfolio construction

Two constructors, both written in numpy/scipy. PyPortfolioOpt is not required.

**Mean-variance.** Maximize \(w^\top\mu - (\lambda/2) w^\top \Sigma w\) with \(\lambda=3\), \(0 \le w_i \le 0.25\), \(\sum w_i \le 1\). Cash holds the residual. \(\mu\) is the LSTM (or Ridge) period forecast annualized by \(252/H\). The covariance used for the ML strategies is \(\Sigma_{ij}=\rho_{ij}\hat\sigma_i\hat\sigma_j\), where \(\rho\) comes from a Ledoit–Wolf shrinkage of trailing 252 daily log returns and \(\hat\sigma\) is the forecast daily vol. If the solved annualized volatility exceeds the target (8 / 12 / 18 percent for low / medium / high), weights are scaled toward cash. There is no leverage.

**Hierarchical risk parity.** Distance \(\sqrt{(1-\rho)/2}\), single linkage, seriation of the dendrogram, recursive inverse-variance bisection, then the same long-only and max-weight projection. This follows the procedure in López de Prado (2016, 2018). It is not a copy of a GPL library.

**Benchmarks.** Equal weight (1/N, still capped at 25 percent). Buy-and-hold 60/40 in SPY and TLT, traded once at the first rebalance date. Static mean-variance using expanding/trailing historical means and Ledoit–Wolf covariance — no LSTM.

Unit tests check long-only, the 25 percent cap, and that weights plus cash sum to one.

## 5. Rebalancing, costs, regime

Rebalance dates are the last session of each month (or quarter) that exists in the price calendar: 44 dates from 2023-01-31 to 2026-08-28 in this run.

Between rebalances, holdings drift with simple returns. On a rebalance close the portfolio earns that day’s return on the incoming (drifted) weights, then trades to the new target. Turnover is \(\sum_i |w^{new}_i - w^{drifted}_i|\). Cost is \(10\,\mathrm{bp} \times\) turnover, subtracted from that day’s portfolio return. The initial allocation is charged the same way. Cash earns the configured rf, default 0.

Regime filter (extra): if SMA(50) of SPY is below SMA(200), using only the close history through the rebalance date, equity and crypto weights are multiplied by 0.5. The freed mass goes to TLT and GLD up to the 25 percent cap, then to cash. The filter is applied to LSTM+MV and LSTM+HRP only, so the benchmarks remain plain. On 2026-08-28 the overlay was **risk-on** (SMA50 = 753.95, SMA200 = 709.10 on the Yahoo SPY series).

Historical 95 percent VaR and CVaR are sample quantiles of the simulated daily portfolio returns. They are not regulatory capital and they are not a forecast of next-month loss. Annualization of VaR by \(\sqrt{252}\) is stored as a diagnostic in code but the CSV reports the daily figures.

## 6. Results of the FULL-public run

Starting capital $100,000, medium risk (12 percent target vol), monthly rebalance, LSTM forecasts, rf = 0. Path length: 933 daily returns after the first close. Numbers below are copied from `results/tables/metrics.csv`.

| Strategy | Total return | CAGR | Vol | Sharpe | Max DD | Avg turnover | VaR 95d | CVaR 95d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM+MV | 0.6157 | 0.1437 | 0.1481 | 0.949 | −0.1696 | 0.3258 | 0.0146 | 0.0214 |
| LSTM+HRP | 0.7244 | 0.1647 | 0.1131 | 1.358 | −0.1074 | 0.1450 | 0.0106 | 0.0159 |
| Equal-weight | 1.1495 | 0.2388 | 0.1727 | 1.283 | −0.2214 | 0.0705 | 0.0157 | 0.0231 |
| Static MV (hist. mean) | 0.9448 | 0.2046 | 0.2020 | 0.990 | −0.1856 | 0.6109 | 0.0190 | 0.0285 |
| Buy-and-hold 60/40 | 0.5488 | 0.1303 | 0.1159 | 1.078 | −0.1409 | 0.0227 | 0.0113 | 0.0163 |

Equal-weight produced the highest total return and CAGR on this sample. LSTM+HRP produced the highest Sharpe and the smallest maximum drawdown, with realized vol 11.3 percent, close to the 12 percent target. LSTM+MV realized 14.8 percent vol, above target, with more turnover than HRP or equal weight. Static MV had the highest turnover (0.61 average L1 per rebalance) and the highest vol. 60/40 had the lowest total return, as one would expect in a period when equities and gold ran and long bonds did not.

None of these rankings is a general result. 2023–2026 in this Yahoo panel is a risk-on window for US large-cap equity, gold, and (over parts of the sample) crypto. A 12 percent vol target, a 25 percent cap, and a defensive overlay that can cut equity in half are designed to *reduce* risk-asset exposure relative to 1/N. In a bull market that reduction shows up as lower return. That is the intended mechanical effect, not a bug, and it is why equal-weight won on CAGR here.

### Latest recommended allocation (MV + regime)

As-of 2026-08-28, medium target vol, long-only, 25 percent cap, regime risk-on so the overlay did not scale anything down. Copied from `results/tables/latest_weights.csv`:

| Asset | MV+regime | HRP (same date) |
| --- | ---: | ---: |
| AAPL | 22.32% | 11.05% |
| MSFT | 0.00% | 9.28% |
| SPY | 22.32% | 7.41% |
| QQQ | 22.32% | 4.09% |
| GLD | 22.32% | 13.60% |
| TLT | 0.00% | 25.00% |
| IWM | 0.00% | 6.43% |
| EFA | 0.00% | 15.04% |
| BTC-USD | 0.00% | 1.07% |
| ETH-USD | 0.00% | 0.61% |
| Cash | 10.71% | 6.43% residual after the 25% TLT cap |

The MV solution concentrates on the four names with the most attractive forecasted return/vol mix (GLD, AAPL, QQQ, SPY), each at the scaled-down cap after the 12 percent vol haircut, plus 10.7 percent cash. MSFT, TLT, IWM, EFA, and both cryptos receive numerical zeros from MV. HRP, which ignores the mean forecast except for reporting, spreads risk more evenly and loads the 25 percent cap on TLT because of its low vol. These are two different answers to “what should I hold tomorrow”; the CLI prints MV+regime as the recommended book because the risk-tolerance knob is a target-vol mean-variance parameter.

Plain-English summary from the run:

> This allocation targets about 12% annualized volatility (risk tolerance: medium) under a long-only mean-variance rule with a 25% per-name cap. Cash residual is 10.7%.

Figures: `results/figures/equity.png`, `drawdown.png`, `pie.png`.

## 7. Limitations

1. **Sample.** One public Yahoo panel, one window, ten names. A different start date, a bear tape, or a universe without BTC would change the ranking.
2. **Forecast quality.** The LSTM is a tiny baseline. Eight epochs and a 20-day return lag are not a serious attempt to model equity premia. Ridge is available and, for many monthly horizons, may be the more honest model.
3. **Covariance.** Mixing LW correlations with forecast vols is a modeling choice. Residuals of the LSTM are not used to rebuild \(\Sigma\). Static MV uses historical means, which in 2023–2026 overweight recent winners and churn (turnover 0.61).
4. **Costs.** 10 bp per unit L1 is a round number. It is too high for SPY and too low for trading ETH in size. There is no bid–ask, no borrow, no FX, no tax lot.
5. **Regime.** Two moving averages on SPY are a textbook overlay, not a regime-switching model. They happened to be risk-on at the end of this sample.
6. **Look-ahead remaining in the *market* data.** Yahoo revisions and backfilled corporate actions can still leak relative to a true point-in-time tape. The software does not leak future *returns* into its own features; that is a weaker guarantee than a vendor point-in-time database.
7. **Target vol vs realized vol.** LSTM+MV was asked for 12 percent and delivered 14.8 percent. The ex-ante constraint is not a realized-vol guarantee.
8. **No live PnL.** There is no broker connection. The latest weights are a research recommendation computed at the last panel date.

## 8. Reproducibility

```text
python -m src.app --config configs/config.yaml --capital 100000 --risk medium --rebalance monthly
python -m pytest -q
```

Seed 42. PyTorch 2.13 CPU. Tests do not retrain the LSTM (`APM_SKIP_TORCH=1` in CI). Transformer remains optional and unused in the reported run.

References used conceptually: Markowitz (1952); Ledoit and Wolf (2004), “Honey, I shrunk the sample covariance matrix”; López de Prado (2016), “Building diversified portfolios that outperform out of sample,” and *Advances in Financial Machine Learning* (2018), ch. 16, for HRP.
