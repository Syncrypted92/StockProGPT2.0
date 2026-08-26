"""Unit tests for StockPro core modules (no Alpaca credentials required)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from stockpro.backtest import run_walk_forward_backtest
from stockpro.broker.live import assert_live_allowed, live_risk_overrides
from stockpro.config import Settings, load_settings
from stockpro.features import FEATURE_COLUMNS, add_forward_labels, add_indicators, feature_matrix
from stockpro.models import chronological_split, train_directional_model
from stockpro.options import (
    ContractCandidate,
    filter_underlying_liquidity,
    select_contract,
    synthetic_candidates_for_backtest,
)
from stockpro.risk import RiskManager, RiskState
from stockpro.signals import Signal, score_to_signal


def _make_ohlcv(n: int = 300, seed: int = 0, start_price: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.015, size=n)
    close = start_price * np.cumprod(1 + rets)
    high = close * (1 + rng.uniform(0.001, 0.01, size=n))
    low = close * (1 - rng.uniform(0.001, 0.01, size=n))
    open_ = close * (1 + rng.normal(0, 0.002, size=n))
    volume = rng.integers(3_000_000, 20_000_000, size=n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "Adj Close": close,
        },
        index=idx,
    )


def test_load_settings():
    s = load_settings()
    assert "universe" in s.raw
    assert s.paper is True or isinstance(s.paper, bool)


def test_indicators_and_labels():
    df = add_forward_labels(add_indicators(_make_ohlcv()), horizon=5)
    assert "RSI" in df.columns
    assert "direction" in df.columns
    X, y = feature_matrix(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y)
    assert set(y.unique()).issubset({-1, 0, 1})


def test_chronological_split_no_shuffle():
    df = add_forward_labels(add_indicators(_make_ohlcv(400)))
    X, y = feature_matrix(df)
    assert len(X) > 50
    Xtr, ytr, Xv, yv, Xte, yte = chronological_split(X, y, 0.7, 0.15)
    assert Xtr.index.max() <= Xv.index.min()
    assert Xv.index.max() <= Xte.index.min()


def test_train_directional_model():
    frames = {
        "SPY": _make_ohlcv(450, seed=0),
        "AAA": _make_ohlcv(450, seed=1),
        "BBB": _make_ohlcv(450, seed=2),
    }
    model, result, clean = train_directional_model(frames)
    assert result.n_train > 0
    assert result.n_test > 0
    row = clean.dropna(subset=FEATURE_COLUMNS).iloc[[-1]][FEATURE_COLUMNS]
    sig = score_to_signal(model, row, "AAA", probability_threshold=0.0)
    assert isinstance(sig.signal, Signal)


def test_liquidity_filter_pass_and_fail():
    good = _make_ohlcv(40, start_price=150)
    report = filter_underlying_liquidity("SPY", good, min_avg_dollar_volume=1_000)
    assert report.passed

    bad = good.copy()
    bad["Volume"] = 100
    report2 = filter_underlying_liquidity("PENNY", bad, min_avg_volume=1_000_000)
    assert not report2.passed
    assert report2.reasons


def test_select_contract_filters_wide_spread():
    spot = 100.0
    today = date.today()
    wide = ContractCandidate(
        symbol="WIDE",
        underlying="SPY",
        option_type="call",
        strike=105,
        expiration=today + timedelta(days=30),
        dte=30,
        bid=1.0,
        ask=2.0,
        mid=1.5,
        spread_pct=0.66,
        open_interest=5000,
        volume=1000,
        delta=0.4,
    )
    tight = ContractCandidate(
        symbol="TIGHT",
        underlying="SPY",
        option_type="call",
        strike=105,
        expiration=today + timedelta(days=30),
        dte=30,
        bid=1.48,
        ask=1.52,
        mid=1.5,
        spread_pct=0.026,
        open_interest=5000,
        volume=1000,
        delta=0.4,
    )
    chosen, notes = select_contract([wide, tight], spot=spot, max_spread_pct=0.08)
    assert chosen is not None
    assert chosen.symbol == "TIGHT"


def test_select_contract_atm_rejects_penny_lottery():
    """0DTE ranking must prefer near-ATM over cheapest far OTM."""
    today = date.today()
    spot = 758.0
    penny = ContractCandidate(
        symbol="PENNY766",
        underlying="SPY",
        option_type="call",
        strike=766,
        expiration=today,
        dte=0,
        bid=0.01,
        ask=0.02,
        mid=0.015,
        spread_pct=0.66,
        open_interest=5000,
        volume=500,
        delta=0.08,
    )
    atm = ContractCandidate(
        symbol="ATM758",
        underlying="SPY",
        option_type="call",
        strike=758,
        expiration=today,
        dte=0,
        bid=1.40,
        ask=1.50,
        mid=1.45,
        spread_pct=0.069,
        open_interest=8000,
        volume=20000,
        delta=0.42,
    )
    # Cheap rank could still pick garbage if spreads/filters pass; atm + floors should not
    chosen, _ = select_contract(
        [penny, atm],
        spot=spot,
        min_dte=0,
        max_dte=0,
        otm_pct_min=-0.003,
        otm_pct_max=0.006,
        min_open_interest=0,
        min_option_volume=0,
        max_spread_pct=0.30,
        min_mid_price=0.40,
        max_mid_price=3.0,
        target_delta_min=0.35,
        target_delta_max=0.55,
        target_delta=0.42,
        rank="atm",
    )
    assert chosen is not None
    assert chosen.symbol == "ATM758"


def test_synthetic_contract_selection():
    cands = synthetic_candidates_for_backtest("SPY", 500.0, "call", date.today())
    chosen, _ = select_contract(cands, spot=500.0)
    assert chosen is not None


def test_risk_sizing_and_daily_halt():
    settings = load_settings()
    rm = RiskManager(settings, RiskState())
    decision = rm.size_order(equity=100_000, premium=1.0)
    assert decision.allowed
    assert decision.qty == 1

    # Blow daily loss limit
    rm.state.daily_pnl = -3000
    ok, reasons = rm.check_loss_limits(100_000)
    assert not ok
    assert reasons


def test_risk_weekly_halt_sets_manual_flag():
    settings = load_settings()
    rm = RiskManager(settings, RiskState())
    rm.state.weekly_pnl = -6000
    ok, reasons = rm.check_loss_limits(100_000)
    assert not ok
    assert rm.state.halted_until_manual


def test_exit_rules():
    settings = load_settings()
    rm = RiskManager(settings)
    exp = date.today() + timedelta(days=30)
    assert rm.should_exit_long(1.0, 1.6, exp, False, profit_target_pct=0.50)[0]  # profit
    assert rm.should_exit_long(1.0, 0.5, exp, False, stop_loss_pct=0.40)[0]  # stop
    assert rm.should_exit_long(1.0, 1.0, date.today() + timedelta(days=0), False, exit_days_before_expiry=0)[0]
    assert rm.should_exit_long(1.0, 1.0, exp, True)[1] == "signal_flip"
    # disabled time stop
    assert rm.should_exit_long(1.0, 1.0, date.today(), False, exit_days_before_expiry=-1) == (False, "")


def test_amd_call_swing_trail():
    from stockpro.risk import amd_swing_signal, option_right, should_exit_amd_call_swing

    assert option_right("SPY260814C00772000") == "call"
    assert option_right("SPY260814P00772000") == "put"
    # Signal: put never; I1 = any call at arm; C5 needs morning+OR
    assert amd_swing_signal(right="put", ret=0.50) == (False, ["put_hard_tp"])
    assert amd_swing_signal(right="call", ret=0.20)[0] is False
    assert amd_swing_signal(right="call", ret=0.40, gate="i1") == (True, ["I1_call", "I1_swing"])
    # Afternoon call still swings on I1
    assert amd_swing_signal(right="call", ret=0.40, arm_hour_et=13.0, or_extension=0.002, gate="i1")[0] is True
    # C5 still requires morning + OR ext
    assert amd_swing_signal(right="call", ret=0.40, arm_hour_et=10.5, or_extension=0.002, gate="c5")[0] is True
    assert "C5_call_morning_ext" in amd_swing_signal(
        right="call", ret=0.40, arm_hour_et=10.5, or_extension=0.002, gate="c5"
    )[1]
    assert amd_swing_signal(right="call", ret=0.40, arm_hour_et=13.0, or_extension=0.002, gate="c5")[0] is False
    assert amd_swing_signal(right="call", ret=0.40, arm_hour_et=10.0, or_extension=0.0005, gate="c5")[0] is False
    # Before arm: hold
    should, reason, peak = should_exit_amd_call_swing(1.0, 1.20, peak_ret=0.0)
    assert not should and peak == pytest.approx(0.20)
    # Armed, still climbing: hold, peak updates
    should, reason, peak = should_exit_amd_call_swing(1.0, 1.70, peak_ret=0.35)
    assert not should and peak == pytest.approx(0.70)
    # Give back 20% from peak 0.70 → exit at ~0.50
    should, reason, peak = should_exit_amd_call_swing(1.0, 1.49, peak_ret=0.70)
    assert should and reason == "trail_stop"
    # Hard SL still works
    should, reason, _ = should_exit_amd_call_swing(1.0, 0.65, peak_ret=0.40)
    assert should and reason == "stop_loss"
    # Runner cap
    should, reason, _ = should_exit_amd_call_swing(1.0, 2.60, peak_ret=1.0)
    assert should and reason == "runner_cap"


def test_scale_out_actions():
    from stockpro.risk import ScaleOutState, next_scale_out_action

    st = ScaleOutState(original_qty=3)
    q, r, st = next_scale_out_action(3, -0.26, tp1_pct=0.30, tp2_pct=0.60, stop_loss_pct=0.25, state=st)
    assert q == 3 and r == "stop_loss"

    st = ScaleOutState(original_qty=3)
    q, r, st = next_scale_out_action(3, 0.31, tp1_pct=0.30, tp2_pct=0.60, stop_loss_pct=0.25, state=st)
    assert q == 1 and r == "profit_target" and st.tp1_done

    q, r, st = next_scale_out_action(2, 0.20, tp1_pct=0.30, tp2_pct=0.60, stop_loss_pct=0.25, state=st)
    assert q == 0

    q, r, st = next_scale_out_action(2, 0.61, tp1_pct=0.30, tp2_pct=0.60, stop_loss_pct=0.25, state=st)
    assert q == 1 and r == "profit_target_2" and st.tp2_done

    q, r, st = next_scale_out_action(1, -0.01, tp1_pct=0.30, tp2_pct=0.60, stop_loss_pct=0.25, state=st)
    assert q == 1 and r == "breakeven"

    st = ScaleOutState(original_qty=3, tp1_done=True)
    q, r, st = next_scale_out_action(2, -0.01, tp1_pct=0.30, tp2_pct=0.60, stop_loss_pct=0.25, state=st)
    assert q == 1 and r == "breakeven" and st.be_done
    q, r, st = next_scale_out_action(1, -0.01, tp1_pct=0.30, tp2_pct=0.60, stop_loss_pct=0.25, state=st)
    assert q == 0  # remaining lot 2 keeps original SL
    q, r, _ = next_scale_out_action(1, -0.26, tp1_pct=0.30, tp2_pct=0.60, stop_loss_pct=0.25, state=st)
    assert q == 1 and r == "stop_loss"

    st = ScaleOutState(original_qty=3)
    q, r, _ = next_scale_out_action(
        3, 0.10, tp1_pct=0.30, tp2_pct=0.60, stop_loss_pct=0.25, state=st, time_flat=True
    )
    assert q == 3 and r == "session_flat"

    st = ScaleOutState(original_qty=3, tp1_done=True, tp2_done=True, peak_ret=0.70)
    q, r, st = next_scale_out_action(
        1, 0.70, tp1_pct=0.30, tp2_pct=0.60, stop_loss_pct=0.25, state=st, runner_trail_pct=0.12
    )
    assert q == 0 and st.peak_ret == 0.70
    q, r, st = next_scale_out_action(
        1, 0.55, tp1_pct=0.30, tp2_pct=0.60, stop_loss_pct=0.25, state=st, runner_trail_pct=0.12
    )
    assert q == 1 and r == "runner_trail"


def test_scale_out_runner_trail_in_sim():
    from datetime import time as dtime

    from stockpro.spy_day.backtest import ScaleOutConfig, simulate_scale_out

    idx = pd.date_range("2026-01-05 10:00", periods=12, freq="5min", tz="America/New_York")
    closes = pd.Series(
        [100.0, 100.4, 100.9, 101.2, 101.5, 101.8, 102.2, 101.6, 101.1, 100.7, 100.3, 99.9],
        index=idx,
    )
    _, _, _, legs = simulate_scale_out(
        closes,
        idx[0],
        1.00,
        100.0,
        1,
        profit_target_pct=0.30,
        stop_loss_pct=0.25,
        short_dte=False,
        force_flat=dtime(15, 45),
        spread_penalty_pct=0.0,
        scale=ScaleOutConfig(qty=3, tp2_pct=0.60, runner_trail_pct=0.20, runner_stop_at_entry=True),
    )
    reasons = [lg["exit_reason"] for lg in legs]
    assert "profit_target" in reasons
    assert any(r in {"runner_trail", "breakeven", "profit_target_2", "force_flat", "eod"} for r in reasons)



def test_live_guards():
    s = Settings(raw={}, paper=True, allow_live=False)
    with pytest.raises(RuntimeError):
        assert_live_allowed(s)
    overrides = live_risk_overrides()
    assert overrides["max_contracts_per_trade"] == 1


def test_backtest_runs():
    frames = {
        "AAA": _make_ohlcv(500, seed=3),
        "BBB": _make_ohlcv(500, seed=4),
    }
    result = run_walk_forward_backtest(
        frames,
        train_days=120,
        test_days=30,
        initial_equity=100_000,
    )
    assert "final_equity" in result.metrics
    assert result.metrics["n_trades"] >= 0
