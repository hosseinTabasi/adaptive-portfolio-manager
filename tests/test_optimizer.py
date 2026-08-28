"""Optimizer constraints: long-only, max weight, weights + cash = 1."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimizer import (
    apply_box_constraints,
    equal_weight,
    hierarchical_risk_parity,
    ledoit_wolf_cov,
    mean_variance,
)


def _spd(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return a @ a.T / n + 0.02 * np.eye(n)


def test_box_constraints_long_only_max_and_budget():
    rng = np.random.default_rng(1)
    raw = rng.normal(size=8)
    w, cash = apply_box_constraints(raw, max_weight=0.25, min_weight=0.0)
    assert np.all(w >= -1e-12)
    assert np.all(w <= 0.25 + 1e-12)
    assert w.sum() + cash == pytest.approx(1.0, abs=1e-8)
    assert cash >= -1e-12


def test_mean_variance_constraints():
    names = [f"A{i}" for i in range(8)]
    mu = pd.Series(np.linspace(0.02, 0.12, 8), index=names)
    cov = pd.DataFrame(_spd(8) / 252.0, index=names, columns=names)
    alloc = mean_variance(mu, cov, max_weight=0.25, min_weight=0.0, risk_aversion=3.0, target_vol=0.12)
    w = alloc.weights.to_numpy()
    assert np.all(w >= -1e-10)
    assert np.all(w <= 0.25 + 1e-10)
    assert w.sum() + alloc.cash == pytest.approx(1.0, abs=1e-7)


def test_hrp_constraints_and_budget():
    names = [f"A{i}" for i in range(10)]
    cov = pd.DataFrame(_spd(10, seed=2) / 252.0, index=names, columns=names)
    alloc = hierarchical_risk_parity(cov, max_weight=0.25)
    w = alloc.weights.to_numpy()
    assert np.all(w >= -1e-10)
    assert np.all(w <= 0.25 + 1e-10)
    assert w.sum() + alloc.cash == pytest.approx(1.0, abs=1e-7)
    # HRP should be fully invested when 10 * 0.25 >= 1 and no min weight
    assert alloc.cash < 0.5


def test_equal_weight_sums_to_one():
    names = ["A", "B", "C", "D"]
    alloc = equal_weight(names, max_weight=0.25)
    assert alloc.weights.sum() + alloc.cash == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(alloc.weights.values, 0.25)


def test_ledoit_wolf_psd():
    rng = np.random.default_rng(3)
    r = pd.DataFrame(rng.normal(0, 0.01, size=(300, 6)), columns=list("ABCDEF"))
    cov = ledoit_wolf_cov(r)
    eig = np.linalg.eigvalsh(cov.to_numpy())
    assert eig.min() > -1e-10
