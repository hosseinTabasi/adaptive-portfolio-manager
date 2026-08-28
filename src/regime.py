"""Simple moving-average regime filter.

If the fast MA of a broad index (default SPY) is below the slow MA, the
filter is "risk-off": equity and crypto weights are scaled down and the
freed capital is moved toward listed defensive names (TLT/GLD) or cash.

Only prices known on the rebalance date are used (inclusive of that close).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data_loader import CRYPTO_LIKE, DEFENSIVE_LIKE, EQUITY_LIKE


@dataclass
class RegimeState:
    bear: bool
    fast_ma: float | None
    slow_ma: float | None
    index: str
    note: str


def moving_average_regime(
    prices: pd.DataFrame,
    as_of,
    *,
    index: str = "SPY",
    fast: int = 50,
    slow: int = 200,
) -> RegimeState:
    """Return bear=True iff SMA_fast < SMA_slow using data through as_of."""
    if index not in prices.columns:
        # first equity-like name, else first column
        candidates = [c for c in prices.columns if c in EQUITY_LIKE] or list(prices.columns)
        index = candidates[0]
        note_prefix = f"Regime index missing; using {index}. "
    else:
        note_prefix = ""
    series = prices[index].loc[:as_of].dropna()
    if series.size < slow:
        return RegimeState(
            bear=False,
            fast_ma=None,
            slow_ma=None,
            index=index,
            note=note_prefix + f"Not enough history for {slow}d MA ({series.size} obs); filter off.",
        )
    sma_f = float(series.iloc[-fast:].mean())
    sma_s = float(series.iloc[-slow:].mean())
    bear = sma_f < sma_s
    note = (
        note_prefix
        + f"{index} SMA{fast}={sma_f:.2f} vs SMA{slow}={sma_s:.2f} "
        + ("risk-off" if bear else "risk-on")
        + f" on {pd.Timestamp(as_of).date()}."
    )
    return RegimeState(bear=bear, fast_ma=sma_f, slow_ma=sma_s, index=index, note=note)


def apply_regime_filter(
    weights: pd.Series,
    prices: pd.DataFrame,
    as_of,
    *,
    enabled: bool = True,
    index: str = "SPY",
    fast: int = 50,
    slow: int = 200,
    risk_scale: float = 0.5,
    equity_tickers: list[str] | None = None,
    crypto_tickers: list[str] | None = None,
    defensive_tickers: list[str] | None = None,
    max_weight: float = 0.25,
) -> tuple[pd.Series, float, RegimeState]:
    """Scale risk assets in a bear regime. Returns (weights, cash, state).

    Cash is 1 - sum(weights) after the adjustment. Defensive names receive
    freed capital pro-rata (then capped at max_weight); leftover stays cash.
    """
    w = weights.astype(float).copy()
    cash = float(max(0.0, 1.0 - w.sum()))
    if not enabled:
        return w, cash, RegimeState(False, None, None, index, "Regime filter disabled.")

    state = moving_average_regime(prices, as_of, index=index, fast=fast, slow=slow)
    if not state.bear:
        return w, cash, state

    equity = set(equity_tickers) if equity_tickers is not None else set(EQUITY_LIKE)
    crypto = set(crypto_tickers) if crypto_tickers is not None else set(CRYPTO_LIKE)
    defensive = set(defensive_tickers) if defensive_tickers is not None else set(DEFENSIVE_LIKE)
    risk = [c for c in w.index if c in equity or c in crypto]
    defs = [c for c in w.index if c in defensive]

    freed = 0.0
    scale = float(min(max(risk_scale, 0.0), 1.0))
    for c in risk:
        old = float(w[c])
        new = old * scale
        w[c] = new
        freed += old - new

    if defs and freed > 0:
        current_def = w[defs].clip(lower=0.0)
        if current_def.sum() <= 1e-12:
            alloc = pd.Series(freed / len(defs), index=defs)
        else:
            alloc = current_def / current_def.sum() * freed
        for c in defs:
            proposed = float(w[c] + alloc[c])
            capped = min(proposed, max_weight)
            leftover = proposed - capped
            w[c] = capped
            freed_back = leftover
            freed = freed - alloc[c] + freed_back
        # any remaining freed that could not go into defensive (caps) becomes cash
        cash = float(max(0.0, 1.0 - w.sum()))
    else:
        cash = float(max(0.0, 1.0 - w.sum()))

    # numerical cleanup
    w = w.clip(lower=0.0)
    if w.sum() + cash > 1 + 1e-8:
        cash = float(max(0.0, 1.0 - w.sum()))
    state.note += f" Scaled {len(risk)} risk names by {scale:.2f}; cash={cash:.3f}."
    return w, cash, state
