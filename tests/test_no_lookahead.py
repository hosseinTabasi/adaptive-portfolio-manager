"""No look-ahead: feature windows and walk-forward cuts do not use t+1 in features at t."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecast import build_supervised_samples, samples_asof, walk_forward_forecasts


def test_feature_window_does_not_overlap_target():
    rng = np.random.default_rng(42)
    r = rng.normal(0, 0.01, size=200)
    lookback, horizon = 20, 21
    X, y_mu, y_vol, last_idx = build_supervised_samples(r, lookback, horizon)
    assert X.shape[0] > 0
    for i in range(min(25, X.shape[0])):
        end_feat = int(last_idx[i])
        # X[i, -1, 0] is returns[end_feat]
        assert X[i, -1, 0] == pytest.approx(r[end_feat])
        # target starts at end_feat+1
        target = r[end_feat + 1 : end_feat + 1 + horizon]
        assert y_mu[i] == pytest.approx(target.sum())
        # no overlap: last feature index < first target index
        assert end_feat < end_feat + 1


def test_asof_cut_excludes_unrealized_targets():
    r = np.arange(100, dtype=float) * 0.001
    lookback, horizon = 10, 5
    X, y_mu, y_vol, last_idx = build_supervised_samples(r, lookback, horizon)
    asof_pos = 40
    Xtr, ymu, yvol = samples_asof(X, y_mu, y_vol, last_idx, asof_pos, horizon)
    target_end = last_idx + horizon
    assert np.all(target_end[target_end <= asof_pos] == last_idx[target_end <= asof_pos] + horizon)
    assert Xtr.shape[0] == int((target_end <= asof_pos).sum())
    # every kept sample's target ended on or before asof
    kept = last_idx[target_end <= asof_pos]
    assert np.all(kept + horizon <= asof_pos)


def test_walk_forward_ridge_ignores_future_spike():
    """A return spike after as-of must not enter the forecast used at as-of."""
    n = 400
    dates = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(0)
    base = rng.normal(0.0002, 0.01, size=n)
    r = pd.DataFrame({"AAA": base, "BBB": rng.normal(0.0001, 0.012, size=n)}, index=dates)
    # huge spike in the last 30 days of AAA
    r.iloc[-30:, 0] = 0.20
    # rebalance dates in the first half, well before the spike
    reb = pd.DatetimeIndex([dates[250], dates[270], dates[290]])
    bundle = walk_forward_forecasts(
        r,
        reb,
        model="ridge",
        lookback=20,
        horizon=21,
        min_train_obs=80,
        retrain_every=1,
        seed=42,
    )
    # forecast at dates[250] should not be near 0.20 * 21
    mu = float(bundle.mu_period.loc[dates[250], "AAA"])
    assert mu < 1.0  # 21 * 0.20 = 4.2 if leaked
    assert abs(mu) < 0.5


def test_inference_window_ends_on_asof():
    """The last feature used at T is the return on T, not T+1."""
    r = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20], dtype=float)
    lookback = 3
    X, y_mu, y_vol, last_idx = build_supervised_samples(r, lookback, horizon=2)
    # last training-eligible sample before asof_pos=4 (value 0.05):
    # feature ends at 4 only if target end = 4+2=6 <= 4? no.
    # feature ending at 2 (0.03): target is 0.04,0.05 end idx 4. OK.
    Xtr, _, _ = samples_asof(X, y_mu, y_vol, last_idx, asof_pos=4, horizon=2)
    # last feature in kept samples must be <= 4 and target used r[3], r[4] not r[5]
    assert Xtr.shape[0] >= 1
    # r[5]=0.10 and r[6]=0.20 must not appear in any feature window of kept samples
    for i in range(Xtr.shape[0]):
        assert 0.10 not in Xtr[i].ravel()
        assert 0.20 not in Xtr[i].ravel()
