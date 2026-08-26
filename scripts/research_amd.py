"""Backtest AMD (Accumulation-Manipulation-Distribution) vs ORB baseline.

AMD alone first (like ORB research), then variants, then ORB+AMD.
"""

from __future__ import annotations

import json
from datetime import time
from pathlib import Path

import pandas as pd

from stockpro.spy_day.amd import AMD_PARAMS, configure_amd_params
from stockpro.spy_day.backtest import run_spy_day_backtest
from stockpro.spy_day.htf_permission import HtfPermissionConfig
from stockpro.spy_day.session import SpyDayConfig, load_spy_day_config

ROOT = Path(__file__).resolve().parents[1]

VARIANTS = [
    {
        "name": "amd_reclaim_am",
        "params": {
            "require_confirm": False,
            "apply_htf": False,
            "window_start": time(10, 0),
            "window_end": time(12, 0),
            "one_per_day": True,
        },
    },
    {
        "name": "amd_confirm_am",
        "params": {
            "require_confirm": True,
            "apply_htf": False,
            "window_start": time(10, 0),
            "window_end": time(12, 0),
            "one_per_day": True,
        },
    },
    {
        "name": "amd_reclaim_htf",
        "params": {
            "require_confirm": False,
            "apply_htf": True,
            "window_start": time(10, 0),
            "window_end": time(12, 0),
            "one_per_day": True,
        },
    },
    {
        "name": "amd_confirm_htf",
        "params": {
            "require_confirm": True,
            "apply_htf": True,
            "window_start": time(10, 0),
            "window_end": time(12, 0),
            "one_per_day": True,
        },
    },
    {
        "name": "amd_reclaim_wide",
        "params": {
            "require_confirm": False,
            "apply_htf": False,
            "window_start": time(10, 0),
            "window_end": time(14, 0),
            "one_per_day": True,
        },
    },
]


def _cfg(patterns: list[str]) -> SpyDayConfig:
    base = load_spy_day_config()
    htf = HtfPermissionConfig(
        enabled=True,
        skip_4h_counter_trend=True,
        eq_context="none",
        eq_tol_pct=0.0025,
        eq_max_dist_pct=0.004,
        eq_min_touches=2,
    )
    pmc = dict(base.pattern_min_confidence)
    pmc.update({"orb": 0.70, "amd": 0.74, "orb_retest": 0.74, "power_hour": 0.72})
    return SpyDayConfig(
        **{
            **{
                k: v
                for k, v in base.__dict__.items()
                if k
                not in (
                    "htf_permission",
                    "patterns",
                    "pattern_min_confidence",
                    "max_trades_per_day",
                )
            },
            "patterns": patterns,
            "score_threshold": 0.70,
            "pattern_min_confidence": pmc,
            "max_trades_per_day": 1 if patterns == ["amd"] or patterns == ["orb"] else 2,
            "htf_permission": htf,
        }
    )


def _run(bars: pd.DataFrame, patterns: list[str], name: str) -> dict:
    print(f"  running {name} ...", flush=True)
    res = run_spy_day_backtest(bars, cfg=_cfg(patterns))
    m = res.metrics
    n_days = int(res.trades["date"].nunique()) if len(res.trades) and "date" in res.trades.columns else 0
    pf = float(m.get("profit_factor", 0) or 0)
    if pf != pf:
        pf = 0.0
    n = float(m.get("n_trades", 0))
    return {
        "name": name,
        "patterns": patterns,
        "n": n,
        "n_days": n_days,
        "tpd": float(m.get("trades_per_day", 0) or 0),
        "wr": float(m.get("win_rate", 0) or 0),
        "exp": float(m.get("expectancy", 0) or 0),
        "pf": pf,
        "pnl": float(res.trades["pnl"].sum()) if len(res.trades) else 0.0,
        "dd": float(m.get("max_drawdown", 0) or 0),
        "gate": bool(pf >= 1.2 and n >= 40),
    }


def _fmt(r: dict) -> str:
    return (
        f"{r['name']:<28} n={r['n']:.0f} days={r['n_days']:<3} "
        f"WR={r['wr']*100:5.1f}% PF={r['pf']:.2f} exp=${r['exp']:.2f} "
        f"pnl=${r['pnl']:.0f} dd={r['dd']*100:.2f}% gate={r['gate']}"
    )


def _reset_amd_defaults() -> None:
    configure_amd_params(
        {
            "window_start": time(10, 0),
            "window_end": time(12, 0),
            "pierce_atr": 0.15,
            "one_per_day": True,
            "require_confirm": False,
            "apply_htf": False,
            "min_body_frac": 0.35,
        }
    )


def main() -> None:
    bars = pd.read_parquet(ROOT / "data" / "bars" / "SPY_5m.parquet")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("America/New_York")

    rows: list[dict] = []
    print("=== AMD research (alone first, like ORB) ===")
    print(f"bars={len(bars)}  {bars.index.min()} -> {bars.index.max()}\n")

    _reset_amd_defaults()
    rows.append(_run(bars, ["orb"], "0_baseline_orb"))
    print(_fmt(rows[-1]))

    print("\n--- AMD alone (variants) ---")
    for v in VARIANTS:
        _reset_amd_defaults()
        configure_amd_params(v["params"])
        # Keep AMD_PARAMS times as time objects
        for k in ("window_start", "window_end"):
            if k in v["params"]:
                AMD_PARAMS[k] = v["params"][k]
        rows.append(_run(bars, ["amd"], f"1_alone:{v['name']}"))
        print(_fmt(rows[-1]))

    # Best AMD alone by PF with n>=20, then pair with ORB
    alone = [r for r in rows if r["name"].startswith("1_alone:") and r["n"] >= 15]
    best = sorted(alone, key=lambda r: (r["pf"], r["exp"], r["n"]), reverse=True)
    print("\n--- ORB + best AMD variants ---")
    for v in VARIANTS[:3]:  # reclaim_am, confirm_am, reclaim_htf
        _reset_amd_defaults()
        configure_amd_params(v["params"])
        for k in ("window_start", "window_end"):
            if k in v["params"]:
                AMD_PARAMS[k] = v["params"][k]
        rows.append(_run(bars, ["orb", "amd"], f"2_orb+{v['name']}"))
        print(_fmt(rows[-1]))

    print("\n--- Ranked (n>=15) ---")
    for r in sorted([x for x in rows if x["n"] >= 15], key=lambda x: (x["pf"], x["exp"]), reverse=True):
        print(_fmt(r))

    if best:
        print(f"\nBest AMD alone: {best[0]['name']} PF={best[0]['pf']:.2f} WR={best[0]['wr']*100:.1f}%")

    out_csv = ROOT / "artifacts" / "amd_research.csv"
    out_json = ROOT / "artifacts" / "amd_research.json"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_csv}")
    _reset_amd_defaults()


if __name__ == "__main__":
    main()
