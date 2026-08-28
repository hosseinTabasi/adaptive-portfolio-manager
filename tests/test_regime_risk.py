"""Regime filter uses only past prices; VaR is a tail statistic of the path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.regime import apply_regime_filter, moving_average_regime
from src.risk import historical_var_cvar, max_drawdown


def test_regime_bear_when_fast_below_slow():
    idx = pd.bdate_range("2020-01-01", periods=260)
    px = pd.DataFrame({"SPY": np.linspace(200, 100, 260)}, index=idx)
    st = moving_average_regime(px, idx[-1], index="SPY", fast=50, slow=200)
    assert st.bear is True


def test_regime_uses_only_asof_slice():
    idx = pd.bdate_range("2020-01-01", periods=300)
    px = pd.DataFrame({"SPY": np.linspace(100, 200, 300)}, index=idx)
    px.iloc[250:, 0] = np.linspace(200, 50, 50)
    st = moving_average_regime(px, idx[240], index="SPY", fast=50, slow=200)
    assert st.bear is False


def test_apply_regime_reduces_equity():
    idx = pd.bdate_range("2020-01-01", periods=260)
    px = pd.DataFrame(
        {
            "SPY": np.linspace(200, 80, 260),
            "AAPL": np.linspace(150, 70, 260),
            "TLT": np.linspace(80, 90, 260),
        },
        index=idx,
    )
    w = pd.Series({"SPY": 0.4, "AAPL": 0.4, "TLT": 0.2})
    new_w, cash, st = apply_regime_filter(w, px, idx[-1], risk_scale=0.5, max_weight=0.50)
    assert st.bear is True
    assert new_w["SPY"] == pytest.approx(0.20, abs=1e-8)
    assert new_w["AAPL"] == pytest.approx(0.20, abs=1e-8)
    assert new_w["TLT"] >= 0.20 - 1e-8
    assert new_w.sum() + cash == pytest.approx(1.0, abs=1e-8)


def test_var_positive_for_negative_tail():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(-0.001, 0.02, size=500))
    report = historical_var_cvar(r, alpha=0.95)
    assert report.var_95 > 0
    assert report.cvar_95 >= report.var_95 - 1e-12


def test_max_drawdown_negative():
    eq = pd.Series([1.0, 1.2, 0.9, 1.1])
    dd = max_drawdown(eq)
    assert dd < 0
    assert abs(dd - (0.9 / 1.2 - 1.0)) < 1e-12
