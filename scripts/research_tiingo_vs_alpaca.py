"""Compare Alpaca vs Tiingo IEX bars for live stack (ORB+AMD) and 4H sweep.

Research only — does not change live data path or enable new patterns.
Tiingo IEX afterHours typically starts ~08:00 ET (still no full London 03–06).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from stockpro.config import ROOT, load_settings
from stockpro.data import get_spy_5m
from stockpro.data.tiingo_bars import (
    download_tiingo_iex_range,
    filter_rth_et,
    load_tiingo_iex,
    tiingo_iex_path,
)
from stockpro.spy_day.amd import configure_amd_params
from stockpro.spy_day.backtest import run_spy_day_backtest
from stockpro.spy_day.h4_sweep_engulf import collect_h4_sweep_engulf
from stockpro.spy_day.mtf_liquidity import clear_htf_cache
from stockpro.spy_day.patterns import enrich_bars
from stockpro.spy_day.pdh_filter import configure_pdh_params
from stockpro.spy_day.po3 import configure_po3_params
from stockpro.spy_day.session import load_spy_day_config
from stockpro.spy_day.smt import configure_smt_params, clear_smt_cache


def _stats(pnls: list[float]) -> dict:
    if not pnls:
        return {"n": 0, "wr": 0.0, "exp": 0.0, "pf": 0.0, "pnl": 0.0}
    s = pd.Series(pnls, dtype=float)
    wins = float(s[s > 0].sum())
    losses = float((-s[s < 0]).sum())
    pf = (wins / losses) if losses > 1e-12 else (99.0 if wins > 0 else 0.0)
    return {
        "n": int(len(s)),
        "wr": float((s > 0).mean()),
        "exp": float(s.mean()),
        "pf": float(pf),
        "pnl": float(s.sum()),
    }


def _sim_vehicle(df: pd.DataFrame, sigs: list, *, vehicle: str) -> list[dict]:
    from datetime import time

    import numpy as np

    from stockpro.spy_day.backtest import _simulate_0dte_path, _simulate_short_dte_path

    closes = df["Close"]
    trades = []
    force_flat = time(15, 45)
    for sig in sigs:
        ts = sig.ts
        if ts not in closes.index:
            pos = closes.index.searchsorted(ts)
            if pos >= len(closes.index):
                continue
            ts = closes.index[pos]
        spot = float(sig.spot)
        side = 1 if sig.side == "call" else -1
        t = ts.timetz().replace(tzinfo=None) if hasattr(ts, "timetz") else ts.time()
        if vehicle == "0dte":
            if t < time(9, 30) or t >= time(15, 45):
                continue
            entry = float(np.clip(spot * 0.002, 0.30, 3.0)) * 1.02
            exit_prem, reason, _ = _simulate_0dte_path(
                closes,
                ts,
                entry,
                spot,
                side,
                profit_target_pct=0.30,
                stop_loss_pct=0.25,
                force_flat=force_flat,
                spread_penalty_pct=0.04,
            )
        else:
            entry = float(np.clip(spot * 0.008, 0.80, 5.0)) * 1.02
            exit_prem, reason, _ = _simulate_short_dte_path(
                closes,
                ts,
                entry,
                spot,
                side,
                profit_target_pct=0.35,
                stop_loss_pct=0.30,
                max_sessions=3,
                spread_penalty_pct=0.04,
            )
        pnl = (exit_prem - entry) * 100
        trades.append({"pnl": pnl, "session": sig.session, "date": ts.date()})
    return trades


def _align_overlap(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same calendar span for fair compare."""
    start = max(a.index.min(), b.index.min())
    end = min(a.index.max(), b.index.max())
    return a.loc[start:end].copy(), b.loc[start:end].copy()


def _hours(df: pd.DataFrame) -> list[int]:
    return sorted(pd.Series(df.index.hour).unique().tolist())


def _run_stack(label: str, bars_rth: pd.DataFrame, base) -> dict:
    configure_po3_params({"enabled": False})
    configure_smt_params({"enabled": False})
    configure_pdh_params({"enabled": False})
    clear_smt_cache()
    configure_amd_params(base.amd.detector_dict())
    cfg = replace(base, patterns=list(base.patterns))
    r = run_spy_day_backtest(bars_rth, cfg=cfg)
    m = r.metrics
    print(
        f"{label:22s} n={int(m['n_trades']):3d} WR={m['win_rate']*100:5.1f}% "
        f"PF={m['profit_factor']:.2f} pnl=${m['final_equity']-100000:.0f}"
    )
    by = {
        p: {
            "n": int(s["n_trades"]),
            "pf": float(s["profit_factor"]),
            "pnl": float(s["total_pnl"]),
        }
        for p, s in r.by_pattern.items()
    }
    for p, s in sorted(by.items()):
        print(f"  {p:14s} n={s['n']:3d} PF={s['pf']:.2f} pnl=${s['pnl']:.0f}")
    clear_htf_cache()
    return {
        "n": int(m["n_trades"]),
        "wr": float(m["win_rate"]),
        "pf": float(m["profit_factor"]),
        "pnl": float(m["final_equity"] - 100000),
        "by_pattern": by,
    }


def _run_h4(label: str, bars_ext: pd.DataFrame) -> dict:
    enriched = enrich_bars(bars_ext, orb_minutes=30)
    sigs = collect_h4_sweep_engulf(enriched)
    out = {"n_signals": len(sigs), "london": 0, "ny": 0, "vehicles": {}}
    out["london"] = sum(1 for s in sigs if s.session == "london")
    out["ny"] = sum(1 for s in sigs if s.session == "ny")
    print(f"{label:22s} signals={len(sigs)} london={out['london']} ny={out['ny']}")
    for vehicle in ("0dte", "short_dte"):
        trades = _sim_vehicle(enriched, sigs, vehicle=vehicle)
        st = _stats([t["pnl"] for t in trades])
        out["vehicles"][vehicle] = st
        print(
            f"  {vehicle:10s} n={st['n']:3d} WR={st['wr']*100:5.1f}% "
            f"PF={st['pf']:.2f} pnl=${st['pnl']:.0f}"
        )
    return out


def main() -> None:
    settings = load_settings()
    base = load_spy_day_config(settings)

    # Ensure Tiingo research cache
    path = tiingo_iex_path("SPY", after_hours=True)
    if not path.exists():
        # match roughly alpaca history window
        alp = get_spy_5m(settings, refresh=False, rth_only=False)
        start = alp.index.min().date() if len(alp) else None
        download_tiingo_iex_range("SPY", start=start, after_hours=True)
    else:
        print(f"using existing {path}")

    tiingo_ah = load_tiingo_iex("SPY", after_hours=True)
    alp_ext = get_spy_5m(settings, refresh=False, rth_only=False)
    alp_rth = get_spy_5m(settings, refresh=False, rth_only=True)
    tiingo_rth = filter_rth_et(tiingo_ah)

    # overlap windows
    alp_rth_o, tiingo_rth_o = _align_overlap(alp_rth, tiingo_rth)
    alp_ext_o, tiingo_ah_o = _align_overlap(alp_ext, tiingo_ah)

    print("=== Coverage ===")
    print(f"Alpaca EXT  n={len(alp_ext)} hours={_hours(alp_ext)} {alp_ext.index.min().date()}->{alp_ext.index.max().date()}")
    print(f"Tiingo AH   n={len(tiingo_ah)} hours={_hours(tiingo_ah)} {tiingo_ah.index.min().date()}->{tiingo_ah.index.max().date()}")
    print(f"Overlap RTH n_alp={len(alp_rth_o)} n_tiingo={len(tiingo_rth_o)}")
    print("NOTE: Tiingo IEX AH still typically starts ~08:00 ET — London 03-06 not covered without BOATS.\n")

    print("=== Live stack ORB+AMD (RTH overlap) ===")
    stack = {
        "alpaca_rth": _run_stack("Alpaca RTH", alp_rth_o, base),
        "tiingo_rth": _run_stack("Tiingo RTH", tiingo_rth_o, base),
    }

    print("\n=== 4H sweep+engulf (extended overlap) ===")
    h4 = {
        "alpaca_ext": _run_h4("Alpaca EXT", alp_ext_o),
        "tiingo_ah": _run_h4("Tiingo AH", tiingo_ah_o),
    }

    # deltas
    print("\n=== Delta (Tiingo - Alpaca) ===")
    sp = stack["tiingo_rth"]["pf"] - stack["alpaca_rth"]["pf"]
    sn = stack["tiingo_rth"]["n"] - stack["alpaca_rth"]["n"]
    print(f"ORB+AMD stack PF: {sp:+.2f}  n: {sn:+d}")
    for vehicle in ("0dte", "short_dte"):
        a = h4["alpaca_ext"]["vehicles"][vehicle]["pf"]
        t = h4["tiingo_ah"]["vehicles"][vehicle]["pf"]
        print(f"H4 {vehicle} PF: {t-a:+.2f} (alpaca {a:.2f} -> tiingo {t:.2f})")

    out = {
        "coverage": {
            "alpaca_hours": _hours(alp_ext),
            "tiingo_hours": _hours(tiingo_ah),
            "london_note": "Neither Alpaca IEX nor Tiingo IEX AH cover 03:00-06:00 ET; need BOATS for true London.",
        },
        "stack": stack,
        "h4": h4,
    }
    path_json = ROOT / "artifacts" / "tiingo_vs_alpaca_research.json"
    path_json.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {path_json}")
    print("Live path unchanged (still Alpaca bars).")


if __name__ == "__main__":
    main()
