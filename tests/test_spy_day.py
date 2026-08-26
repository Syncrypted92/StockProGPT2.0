"""Tests for SPY day 5m patterns and backtest (no Alpaca required)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockpro.spy_day.backtest import run_spy_day_backtest
from stockpro.spy_day.htf_permission import HtfPermissionConfig
from stockpro.spy_day.patterns import best_signal, configure_htf_permission, enrich_bars
from stockpro.spy_day.session import SpyDayConfig, session_allows_entry


def _make_spy_5m(n_days: int = 5, bars_per_day: int = 78, seed: int = 0) -> pd.DataFrame:
    """Synthetic RTH 5m bars (09:30–15:55)."""
    rng = np.random.default_rng(seed)
    idx = []
    data = []
    price = 500.0
    start = pd.Timestamp("2026-01-05 09:30", tz="America/New_York")
    for d in range(n_days + 3):
        day = start + pd.Timedelta(days=d)
        if day.dayofweek >= 5:
            continue
        if len({x.date() for x in idx}) >= n_days:
            break
        for b in range(bars_per_day):
            ts = day + pd.Timedelta(minutes=5 * b)
            ret = rng.normal(0.00005, 0.0012)
            o = price
            c = price * (1 + ret)
            h = max(o, c) * (1 + abs(rng.normal(0, 0.0004)))
            l = min(o, c) * (1 - abs(rng.normal(0, 0.0004)))
            vol = int(rng.integers(50_000, 200_000))
            data.append({"Open": o, "High": h, "Low": l, "Close": c, "Volume": vol})
            idx.append(ts)
            price = c
    return pd.DataFrame(data, index=pd.DatetimeIndex(idx, name="timestamp"))


def test_enrich_and_detect():
    configure_htf_permission(HtfPermissionConfig(enabled=False))
    df = _make_spy_5m(8)
    enriched = enrich_bars(df, orb_minutes=30)
    assert "VWAP" in enriched.columns
    assert "EMA9" in enriched.columns
    assert "OR_HIGH" in enriched.columns
    _ = best_signal(enriched, min_confidence=0.5)


def test_backtest_runs():
    cfg = SpyDayConfig(
        score_threshold=0.50,
        max_trades_per_day=5,
        htf_permission=HtfPermissionConfig(enabled=False),
    )
    df = _make_spy_5m(12, seed=3)
    result = run_spy_day_backtest(df, cfg=cfg)
    assert "n_trades" in result.metrics
    assert result.metrics["n_trades"] >= 0


def test_session_window():
    ok, reason = session_allows_entry(
        pd.Timestamp("2026-01-06 11:00", tz="America/New_York").to_pydatetime(),
        cfg=SpyDayConfig(),
    )
    assert ok and reason == "ok"
    ok2, reason2 = session_allows_entry(
        pd.Timestamp("2026-01-06 15:00", tz="America/New_York").to_pydatetime(),
        cfg=SpyDayConfig(),
    )
    assert (not ok2) and reason2 == "too_late"


def test_scale_out_three_legs_on_stop():
    from datetime import time as dtime

    from stockpro.spy_day.backtest import ScaleOutConfig, simulate_scale_out

    idx = pd.date_range("2026-01-05 10:00", periods=8, freq="5min", tz="America/New_York")
    # Dump after entry so all lots stop together.
    closes = pd.Series([500.0, 498.0, 496.0, 494.0, 492.0, 490.0, 488.0, 486.0], index=idx)
    pnl, reason, exit_j, legs = simulate_scale_out(
        closes,
        idx[0],
        1.50,
        500.0,
        1,
        profit_target_pct=0.30,
        stop_loss_pct=0.25,
        short_dte=False,
        force_flat=dtime(15, 45),
        spread_penalty_pct=0.04,
        scale=ScaleOutConfig(),
    )
    assert len(legs) == 3
    assert all(lg["exit_reason"] == "stop_loss" for lg in legs)
    assert pnl < 0
    assert exit_j > 0
    assert "stop_loss" in reason


def test_scale_out_two_lots_on_stop():
    from datetime import time as dtime

    from stockpro.spy_day.backtest import ScaleOutConfig, simulate_scale_out

    idx = pd.date_range("2026-01-05 10:00", periods=8, freq="5min", tz="America/New_York")
    closes = pd.Series([500.0, 498.0, 496.0, 494.0, 492.0, 490.0, 488.0, 486.0], index=idx)
    pnl, reason, exit_j, legs = simulate_scale_out(
        closes,
        idx[0],
        1.50,
        500.0,
        1,
        profit_target_pct=0.35,
        stop_loss_pct=0.25,
        short_dte=False,
        force_flat=dtime(15, 45),
        spread_penalty_pct=0.04,
        scale=ScaleOutConfig(qty=2, tp1_pct=0.35, tp2_pct=0.60, runner_stop_at_entry=True),
    )
    assert len(legs) == 2
    assert all(lg["exit_reason"] == "stop_loss" for lg in legs)
    assert pnl < 0
    assert exit_j > 0


def test_scale_out_qty_by_pattern():
    from stockpro.spy_day.session import ScaleOutConfig

    cfg = ScaleOutConfig.from_dict(
        {
            "enabled": True,
            "qty": 3,
            "qty_by_pattern": {"orb": 2, "orb_retest": 2, "power_hour": 3, "amd": 3},
        }
    )
    assert cfg.qty_for("orb") == 2
    assert cfg.qty_for("amd") == 3
    assert cfg.for_pattern("orb").qty == 2


def test_ote_fib_zone():
    from stockpro.spy_day.ote import ImpulseSwing, OTE_END, OTE_START

    bull = ImpulseSwing("bull", 0, 10, 100.0, 110.0, True)
    assert bull.in_ote_zone(bull.fib_level(0.705))
    z0, _, z1 = bull.ote_bounds()
    assert abs(bull.fib_level(OTE_START) - z0) < 1e-9
    assert abs(bull.fib_level(OTE_END) - z1) < 1e-9
    # 50% is outside OTE
    assert not bull.in_ote_zone(bull.fib_level(0.50))
