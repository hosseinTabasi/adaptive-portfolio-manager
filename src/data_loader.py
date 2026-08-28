"""Download, cache, and clean a multi-asset daily price panel.

Yahoo Finance via yfinance is the default public source. Failed tickers are
dropped and recorded. If every ticker fails, a labeled TOY synthetic panel is
returned so the rest of the pipeline can still run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "SPY",
    "QQQ",
    "GLD",
    "TLT",
    "IWM",
    "EFA",
    "BTC-USD",
    "ETH-USD",
]

EQUITY_LIKE = {"AAPL", "MSFT", "SPY", "QQQ", "IWM", "EFA"}
CRYPTO_LIKE = {"BTC-USD", "ETH-USD"}
DEFENSIVE_LIKE = {"TLT", "GLD"}


@dataclass
class PanelResult:
    """Cleaned close prices, log returns, and download provenance."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    tickers: list[str]
    failed_tickers: list[str]
    source: str  # "yahoo" | "toy"
    label: str  # "FULL-public" | "TOY"
    notes: list[str] = field(default_factory=list)
    synthetic_bond_used: bool = False

    def to_meta(self) -> dict:
        return {
            "tickers": self.tickers,
            "failed_tickers": self.failed_tickers,
            "source": self.source,
            "label": self.label,
            "notes": self.notes,
            "synthetic_bond_used": self.synthetic_bond_used,
            "n_obs": int(len(self.prices)),
            "start": str(self.prices.index.min().date()) if len(self.prices) else None,
            "end": str(self.prices.index.max().date()) if len(self.prices) else None,
        }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_toy_panel(
    n_days: int = 1600,
    seed: int = 42,
    start: str = "2021-01-01",
    tickers: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Factor-model synthetic daily closes, labeled TOY in PanelResult.

    A single market factor plus idiosyncratic noise. Bond-like names get
    lower beta and vol; crypto-like names get higher vol. This is not a
    calibrated replica of any listed security.
    """
    rng = np.random.default_rng(seed)
    names = list(tickers) if tickers is not None else list(DEFAULT_TICKERS)
    dates = pd.bdate_range(start=start, periods=n_days)
    market = rng.normal(0.00025, 0.011, size=n_days)
    closes = {}
    for i, name in enumerate(names):
        if name in DEFENSIVE_LIKE or name.upper().startswith("TLT") or "BOND" in name.upper():
            beta, idio, mu = 0.15, 0.0055, 0.00008
        elif name in CRYPTO_LIKE or "BTC" in name.upper() or "ETH" in name.upper():
            beta, idio, mu = 1.4, 0.035, 0.0004
        else:
            beta, idio, mu = 0.9 + 0.05 * (i % 3), 0.012, 0.0003
        r = mu + beta * market + rng.normal(0.0, idio, size=n_days)
        px = 100.0 * np.exp(np.cumsum(r))
        closes[name] = px
    df = pd.DataFrame(closes, index=dates)
    df.index.name = "Date"
    return df


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Handle yfinance multi-ticker MultiIndex and single-ticker frames."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = [str(x) for x in raw.columns.get_level_values(0)]
        level1 = [str(x) for x in raw.columns.get_level_values(1)]
        if "Close" in level0:
            close = raw["Close"].copy()
        elif "Close" in level1:
            close = raw.xs("Close", axis=1, level=1).copy()
        else:
            close = raw.iloc[:, 0:0].copy()
        if isinstance(close, pd.Series):
            close = close.to_frame()
        close.columns = [str(c) for c in close.columns]
        return close
    if "Close" in raw.columns:
        col = raw["Close"].to_frame()
        col.columns = [tickers[0] if tickers else "Close"]
        return col
    return pd.DataFrame(raw)


def download_yahoo(
    tickers: list[str],
    start: str,
    end: str | None,
    cache_dir: Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Fetch adjusted close per ticker. Returns (close, ok, failed)."""
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame(), [], list(tickers)

    _ensure_dir(cache_dir)
    ok: list[str] = []
    failed: list[str] = []
    frames: list[pd.Series] = []

    for t in tickers:
        cache_path = cache_dir / f"{t.replace('/', '-')}.csv"
        series = None
        try:
            raw = yf.download(
                t,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            close = _extract_close(raw, [t])
            if close.empty or close.dropna().empty:
                raise ValueError("empty download")
            col = close.columns[0]
            series = close[col].rename(t).dropna()
            if series.size < 60:
                raise ValueError(f"too few rows ({series.size})")
            series.to_csv(cache_path, header=True)
            ok.append(t)
            frames.append(series)
        except Exception as exc:  # noqa: BLE001 — per-ticker isolation
            # try cache
            if cache_path.exists():
                try:
                    cached = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
                    cached.name = t
                    if cached.dropna().size >= 60:
                        ok.append(t)
                        frames.append(cached)
                        continue
                except Exception:  # noqa: BLE001
                    pass
            failed.append(t)
            print(f"[data] drop {t}: {exc}")

    if not frames:
        return pd.DataFrame(), ok, failed
    prices = pd.concat(frames, axis=1).sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    return prices, ok, failed


def clean_panel(
    prices: pd.DataFrame,
    max_ffill_days: int = 5,
) -> pd.DataFrame:
    """Align on a weekday calendar, limited ffill, drop leading NaNs.

    Crypto trades on weekends; equities do not. The panel is restricted to
    weekdays so that weekend crypto prints do not dilute equity volatility.
    Holiday gaps up to ``max_ffill_days`` are forward-filled. Remaining
    leading NaNs (assets that listed later) are dropped row-wise only until
    every remaining column has at least one observation; columns that are
    still empty are dropped.
    """
    if prices.empty:
        return prices
    px = prices.copy()
    px.index = pd.to_datetime(px.index)
    px = px[~px.index.duplicated(keep="last")].sort_index()
    # weekday calendar spanning the observed range
    bdays = pd.bdate_range(px.index.min(), px.index.max())
    px = px.reindex(px.index.union(bdays)).sort_index()
    px = px.ffill(limit=max_ffill_days)
    px = px.loc[px.index.dayofweek < 5]
    px = px.dropna(axis=1, how="all")
    px = px.dropna(axis=0, how="all")
    # drop rows at the start until each live column is non-NaN
    if not px.empty:
        first_valid = px.apply(lambda s: s.first_valid_index())
        # keep columns with some data
        keep = [c for c in px.columns if first_valid[c] is not None]
        px = px[keep]
        if keep:
            start = max(first_valid[c] for c in keep)
            px = px.loc[start:]
        px = px.ffill(limit=max_ffill_days)
        # remaining interior holes: drop rows that still have any NaN after limited ffill
        px = px.dropna(axis=0, how="any")
    px.index.name = "Date"
    return px


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    rets = np.log(prices / prices.shift(1))
    return rets.dropna(how="all")


def maybe_add_synthetic_bond(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, bool, str]:
    """If TLT is missing, add a low-vol synthetic bond proxy (documented)."""
    if "TLT" in prices.columns:
        return prices, returns, False, ""
    rng = np.random.default_rng(seed)
    n = len(returns)
    if n == 0:
        return prices, returns, False, ""
    if "SPY" in returns.columns:
        mkt = returns["SPY"].fillna(0.0).to_numpy()
    else:
        mkt = returns.mean(axis=1).fillna(0.0).to_numpy()
    # modest positive drift, low vol, small negative equity beta
    r = 0.00008 - 0.12 * mkt + rng.normal(0.0, 0.0045, size=n)
    # align length with prices (returns is one row shorter)
    px0 = 100.0
    # pad a leading zero return so prices has the same index as original prices
    bond_ret_full = pd.Series(0.0, index=prices.index, name="TLT_SYN")
    bond_ret_full.loc[returns.index] = r
    bond_px = px0 * np.exp(bond_ret_full.cumsum())
    prices = prices.copy()
    returns = returns.copy()
    prices["TLT_SYN"] = bond_px
    returns["TLT_SYN"] = log_returns(prices[["TLT_SYN"]])["TLT_SYN"]
    note = (
        "TLT was missing; added TLT_SYN, a synthetic low-vol bond proxy "
        "with small negative equity beta. 60/40 uses SPY/TLT_SYN. TOY component."
    )
    return prices, returns, True, note


def load_panel(
    tickers: Iterable[str] | None = None,
    start: str = "2021-01-01",
    end: str | None = None,
    cache_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
    max_ffill_days: int = 5,
    force_toy: bool = False,
) -> PanelResult:
    """Load a cleaned panel from Yahoo, or TOY if the download fully fails."""
    names = list(tickers) if tickers is not None else list(DEFAULT_TICKERS)
    root = _project_root()
    cache = Path(cache_dir) if cache_dir else root / "data" / "cache"
    processed = Path(processed_dir) if processed_dir else root / "data" / "processed"
    _ensure_dir(cache)
    _ensure_dir(processed)
    notes: list[str] = []

    if force_toy:
        prices_raw = generate_toy_panel(start=start, tickers=names)
        failed: list[str] = []
        ok = list(prices_raw.columns)
        source = "toy"
        label = "TOY"
        notes.append("force_toy=True; synthetic factor-model panel.")
    else:
        prices_raw, ok, failed = download_yahoo(names, start, end, cache)
        if prices_raw.empty or len(ok) == 0:
            notes.append(
                "All Yahoo downloads failed; generated a labeled TOY synthetic panel."
            )
            prices_raw = generate_toy_panel(start=start, tickers=names)
            ok = list(prices_raw.columns)
            failed = list(names)
            source = "toy"
            label = "TOY"
        else:
            source = "yahoo"
            label = "FULL-public"
            if failed:
                notes.append(f"Dropped tickers with failed downloads: {failed}")

    prices = clean_panel(prices_raw, max_ffill_days=max_ffill_days)
    if prices.empty or prices.shape[1] == 0:
        notes.append("Cleaned panel empty; falling back to TOY.")
        prices = clean_panel(generate_toy_panel(start=start, tickers=names), max_ffill_days)
        source = "toy"
        label = "TOY"
        failed = list(names)

    rets = log_returns(prices)
    prices, rets, syn_bond, bond_note = maybe_add_synthetic_bond(prices, rets)
    if bond_note:
        notes.append(bond_note)
        # synthetic bond means the 60/40 leg is not a listed TLT series
        if label == "FULL-public":
            notes.append("Panel is FULL-public except TLT_SYN (TOY bond proxy).")

    result = PanelResult(
        prices=prices,
        returns=rets.reindex(prices.index).fillna(0.0).iloc[1:],
        tickers=list(prices.columns),
        failed_tickers=failed,
        source=source,
        label=label,
        notes=notes,
        synthetic_bond_used=syn_bond,
    )
    # keep a small processed panel on disk
    prices.to_csv(processed / "prices.csv")
    result.returns.to_csv(processed / "returns.csv")
    (processed / "meta.json").write_text(json.dumps(result.to_meta(), indent=2))
    return result
