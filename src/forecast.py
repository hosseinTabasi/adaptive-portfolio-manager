"""Walk-forward return and volatility forecasts.

Features at date t use information available at t only. Supervised targets
are the *next* ``horizon``-day sum of log returns and the *next* ``horizon``-day
realized volatility. A sample is eligible for training at as-of date T only
when its target window ends on or before T.

Default model: a one-layer LSTM (hidden 16). Optional tiny Transformer behind
the config flag ``forecast.model: transformer``. sklearn Ridge is the
documented fallback when PyTorch is unavailable or training is skipped.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

ModelName = Literal["lstm", "ridge", "transformer"]


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_supervised_samples(
    returns: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y_mu, y_vol, last_feature_index) with no look-ahead overlap.

    X[i] = returns[i : i+lookback]
    y_mu[i] = sum(returns[i+lookback : i+lookback+horizon])   # next-period return
    y_vol[i] = std(returns[i+lookback : i+lookback+horizon], ddof=1)
    last_feature_index[i] = i+lookback-1   # index of the last feature return

    The target window starts at i+lookback, so it does not overlap the feature
    window. Callers must still restrict samples by as-of date.
    """
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    n = r.size
    xs: list[np.ndarray] = []
    y_mu: list[float] = []
    y_vol: list[float] = []
    last_idx: list[int] = []
    last_start = n - lookback - horizon
    if last_start < 0:
        empty_x = np.zeros((0, lookback, 1), dtype=np.float64)
        z = np.zeros((0,), dtype=np.float64)
        zi = np.zeros((0,), dtype=np.int64)
        return empty_x, z, z, zi
    for i in range(0, last_start + 1):
        window = r[i : i + lookback]
        target = r[i + lookback : i + lookback + horizon]
        if not np.isfinite(window).all() or not np.isfinite(target).all():
            continue
        if target.size < max(2, horizon // 2):
            continue
        xs.append(window.reshape(lookback, 1))
        y_mu.append(float(target.sum()))
        y_vol.append(float(np.std(target, ddof=1)))
        last_idx.append(i + lookback - 1)
    if not xs:
        empty_x = np.zeros((0, lookback, 1), dtype=np.float64)
        z = np.zeros((0,), dtype=np.float64)
        zi = np.zeros((0,), dtype=np.int64)
        return empty_x, z, z, zi
    return (
        np.stack(xs, axis=0),
        np.asarray(y_mu, dtype=np.float64),
        np.asarray(y_vol, dtype=np.float64),
        np.asarray(last_idx, dtype=np.int64),
    )


def samples_asof(
    X: np.ndarray,
    y_mu: np.ndarray,
    y_vol: np.ndarray,
    last_idx: np.ndarray,
    asof_pos: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep samples whose *target window* ends on or before ``asof_pos``.

    Target end index = last_feature_index + horizon. This is the no-look-ahead
    cut: at close of day T we may train on targets that have already realized.
    """
    if last_idx.size == 0:
        return X, y_mu, y_vol
    target_end = last_idx + horizon
    mask = target_end <= asof_pos
    return X[mask], y_mu[mask], y_vol[mask]


def _torch_available() -> bool:
    if os.environ.get("APM_SKIP_TORCH", "").strip() in {"1", "true", "True"}:
        return False
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


class _RidgePair:
    """Two independent Ridge models: next-period return and daily vol."""

    def __init__(self, alpha: float = 1.0):
        from sklearn.linear_model import Ridge

        self.mu = Ridge(alpha=alpha)
        self.vol = Ridge(alpha=alpha)
        self.fitted = False

    def fit(self, X: np.ndarray, y_mu: np.ndarray, y_vol: np.ndarray) -> None:
        flat = X.reshape(X.shape[0], -1)
        self.mu.fit(flat, y_mu)
        self.vol.fit(flat, y_vol)
        self.fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flat = X.reshape(X.shape[0], -1)
        mu = self.mu.predict(flat)
        vol = np.clip(self.vol.predict(flat), 1e-8, None)
        return mu, vol


def _build_torch_model(name: ModelName, hidden: int, lookback: int, n_layers: int, seed: int):
    import torch
    from torch import nn

    set_seed(seed)

    class LSTMHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden,
                num_layers=n_layers,
                batch_first=True,
            )
            self.head = nn.Linear(hidden, 2)

        def forward(self, x):  # x: (B, T, 1)
            out, _ = self.lstm(x)
            pred = self.head(out[:, -1, :])
            mu = pred[:, 0]
            log_vol = pred[:, 1]
            return mu, log_vol

    class TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            d_model = max(8, hidden)
            nhead = 2 if d_model % 2 == 0 else 1
            self.proj = nn.Linear(1, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=max(16, hidden * 2),
                batch_first=True,
                dropout=0.0,
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=1)
            self.head = nn.Linear(d_model, 2)
            self._lookback = lookback

        def forward(self, x):
            h = self.proj(x)
            h = self.enc(h)
            pred = self.head(h[:, -1, :])
            return pred[:, 0], pred[:, 1]

    if name == "transformer":
        return TinyTransformer()
    return LSTMHead()


def _train_torch(
    name: ModelName,
    X: np.ndarray,
    y_mu: np.ndarray,
    y_vol: np.ndarray,
    *,
    hidden: int,
    n_layers: int,
    lookback: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(seed)
    device = torch.device("cpu")
    model = _build_torch_model(name, hidden, lookback, n_layers, seed).to(device)
    # standardize targets on the training fold only
    mu_mean, mu_std = float(y_mu.mean()), float(y_mu.std() + 1e-8)
    vol_mean, vol_std = float(y_vol.mean()), float(y_vol.std() + 1e-8)
    y_mu_z = (y_mu - mu_mean) / mu_std
    y_vol_z = (y_vol - vol_mean) / vol_std
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.stack(
        [
            torch.tensor(y_mu_z, dtype=torch.float32),
            torch.tensor(y_vol_z, dtype=torch.float32),
        ],
        dim=1,
    )
    loader = DataLoader(
        TensorDataset(xt, yt),
        batch_size=min(batch_size, max(8, len(xt))),
        shuffle=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred_mu, pred_logvol = model(xb)
            # predict standardized vol via exp(log) mapped loosely: use pred_logvol as z-vol
            loss = loss_fn(pred_mu, yb[:, 0]) + loss_fn(pred_logvol, yb[:, 1])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    stats = {"mu_mean": mu_mean, "mu_std": mu_std, "vol_mean": vol_mean, "vol_std": vol_std}
    return model, stats


def _predict_torch(model, stats: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32)
        mu_z, vol_z = model(xt)
        mu = mu_z.cpu().numpy() * stats["mu_std"] + stats["mu_mean"]
        vol = vol_z.cpu().numpy() * stats["vol_std"] + stats["vol_mean"]
    vol = np.clip(vol, 1e-8, None)
    return mu, vol


@dataclass
class ForecastBundle:
    """Period expected returns (not annualized) and daily-vol forecasts."""

    mu_period: pd.DataFrame  # index: rebalance dates, columns: assets
    vol_daily: pd.DataFrame
    model_used: str
    notes: list[str]


def _last_window(returns: np.ndarray, lookback: int) -> np.ndarray | None:
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    if r.size < lookback:
        return None
    w = r[-lookback:]
    if not np.isfinite(w).all():
        return None
    return w.reshape(1, lookback, 1)


def _trailing_fallback(r: np.ndarray, horizon: int) -> tuple[float, float]:
    if r.size < 5:
        return 0.0, 0.01
    mu = float(np.mean(r[-horizon:]) * horizon) if r.size >= horizon else float(np.mean(r) * horizon)
    vol = float(np.std(r[-max(horizon, 20) :], ddof=1))
    return mu, max(vol, 1e-8)


def walk_forward_forecasts(
    returns: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    *,
    model: ModelName = "lstm",
    lookback: int = 20,
    horizon: int = 21,
    hidden_size: int = 16,
    num_layers: int = 1,
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    seed: int = 42,
    retrain_every: int = 4,
    min_train_obs: int = 252,
    models_dir: Path | str | None = None,
) -> ForecastBundle:
    """Expanding-window forecasts at each rebalance date (no look-ahead)."""
    notes: list[str] = []
    use_torch = model in {"lstm", "transformer"} and _torch_available()
    if model in {"lstm", "transformer"} and not use_torch:
        notes.append(
            f"Requested {model} but torch is unavailable or APM_SKIP_TORCH=1; "
            "using sklearn Ridge walk-forward fallback."
        )
        model_used = "ridge"
    else:
        model_used = model if use_torch else "ridge"

    assets = list(returns.columns)
    mu_rows: list[dict] = []
    vol_rows: list[dict] = []
    set_seed(seed)

    # precompute sequences per asset on the full series; slice by as-of in the loop
    seqs: dict[str, tuple] = {}
    for a in assets:
        seqs[a] = build_supervised_samples(returns[a].to_numpy(), lookback, horizon)

    fitted: dict[str, object] = {}
    fitted_kind = model_used
    last_saved_state = None

    for k, dt in enumerate(rebalance_dates):
        hist = returns.loc[:dt]
        if hist.empty:
            continue
        asof_pos = int(returns.index.searchsorted(dt, side="right") - 1)
        if asof_pos < 0:
            continue
        do_train = (k % max(1, retrain_every) == 0) or not fitted
        mu_hat: dict[str, float] = {}
        vol_hat: dict[str, float] = {}

        for a in assets:
            series = hist[a].dropna().to_numpy()
            Xall, ymu_all, yvol_all, last_idx = seqs[a]
            Xtr, ymu, yvol = samples_asof(Xall, ymu_all, yvol_all, last_idx, asof_pos, horizon)
            if do_train:
                if Xtr.shape[0] < 40 or series.size < min_train_obs:
                    fitted[a] = None
                elif fitted_kind == "ridge":
                    m = _RidgePair(alpha=1.0)
                    m.fit(Xtr, ymu, yvol)
                    fitted[a] = m
                else:
                    try:
                        m, stats = _train_torch(
                            fitted_kind,  # type: ignore[arg-type]
                            Xtr,
                            ymu,
                            yvol,
                            hidden=hidden_size,
                            n_layers=num_layers,
                            lookback=lookback,
                            epochs=epochs,
                            batch_size=batch_size,
                            lr=learning_rate,
                            seed=seed,
                        )
                        fitted[a] = (m, stats)
                    except Exception as exc:  # noqa: BLE001
                        notes.append(f"{a} at {dt.date()}: torch train failed ({exc}); Ridge.")
                        m = _RidgePair(alpha=1.0)
                        m.fit(Xtr, ymu, yvol)
                        fitted[a] = m

            x_inf = _last_window(series, lookback)
            obj = fitted.get(a)
            if obj is None or x_inf is None:
                mu_i, vol_i = _trailing_fallback(series, horizon)
            elif isinstance(obj, _RidgePair):
                mu_p, vol_p = obj.predict(x_inf)
                mu_i, vol_i = float(mu_p[0]), float(vol_p[0])
            else:
                m, stats = obj
                mu_p, vol_p = _predict_torch(m, stats, x_inf)
                mu_i, vol_i = float(mu_p[0]), float(vol_p[0])
            mu_hat[a] = mu_i
            vol_hat[a] = max(vol_i, 1e-8)

        mu_hat["date"] = dt
        vol_hat["date"] = dt
        mu_rows.append(mu_hat)
        vol_rows.append(vol_hat)

        if do_train and fitted_kind in {"lstm", "transformer"}:
            last_saved_state = (dt, dict(fitted))

    mu_df = pd.DataFrame(mu_rows).set_index("date")
    vol_df = pd.DataFrame(vol_rows).set_index("date")

    if models_dir is not None and last_saved_state is not None:
        _save_last_models(Path(models_dir), last_saved_state[1], fitted_kind, seed, lookback, hidden_size)
        notes.append(f"Saved last {fitted_kind} weights under {models_dir} (as-of {last_saved_state[0].date()}).")
    elif models_dir is not None and fitted_kind == "ridge":
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        meta = {"model": "ridge", "seed": seed, "lookback": lookback, "horizon": horizon}
        (Path(models_dir) / "metadata.json").write_text(json.dumps(meta, indent=2))

    return ForecastBundle(mu_period=mu_df, vol_daily=vol_df, model_used=fitted_kind, notes=notes)


def _save_last_models(models_dir: Path, fitted: dict, kind: str, seed: int, lookback: int, hidden: int) -> None:
    import torch

    models_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for asset, obj in fitted.items():
        if obj is None or isinstance(obj, _RidgePair):
            continue
        model, stats = obj
        safe = asset.replace("/", "-")
        path = models_dir / f"{kind}_{safe}.pt"
        torch.save({"state_dict": model.state_dict(), "stats": stats, "asset": asset}, path)
        saved.append(safe)
    meta = {
        "model": kind,
        "seed": seed,
        "lookback": lookback,
        "hidden_size": hidden,
        "assets_saved": saved,
    }
    (models_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def load_model_state(models_dir: Path | str, asset: str, kind: str = "lstm"):
    """Load a previously saved state_dict bundle, or None."""
    import torch

    safe = asset.replace("/", "-")
    path = Path(models_dir) / f"{kind}_{safe}.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)
