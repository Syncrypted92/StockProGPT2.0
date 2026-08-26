"""Backtest new intraday strategies alone and paired with ORB.

Keeps ORB as baseline. Reports WR / PF / trades / gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stockpro.spy_day.backtest import run_spy_day_backtest
from stockpro.spy_day.htf_permission import HtfPermissionConfig
from stockpro.spy_day.session import SpyDayConfig, load_spy_day_config

ROOT = Path(__file__).resolve().parents[1]

STRATEGIES = [
    "gap_and_go",
    "opening_drive",
    "trend_day_pullback",
    "failed_auction",
    "range_day_fade",
    "power_hour",
    "multi_day_break",
    "compression_break",
    "spy_qqq_lead",
]

PMC = {
    "orb": 0.70,
    "orb_retest": 0.74,
    "gap_and_go": 0.74,
    "opening_drive": 0.74,
    "trend_day_pullback": 0.74,
    "failed_auction": 0.74,
    "range_day_fade": 0.72,
    "power_hour": 0.72,
    "multi_day_break": 0.74,
    "compression_break": 0.74,
    "spy_qqq_lead": 0.74,
}


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
    return SpyDayConfig(
        **{
            **{
                k: v
                for k, v in base.__dict__.items()
                if k not in ("htf_permission", "patterns", "pattern_min_confidence", "max_trades_per_day")
            },
            "patterns": patterns,
            "score_threshold": 0.70,
            "pattern_min_confidence": PMC,
            "max_trades_per_day": 2,
            "htf_permission": htf,
        }
    )


def _run(bars: pd.DataFrame, patterns: list[str], name: str) -> dict:
    from stockpro.spy_day.intraday_patterns import clear_intraday_caches

    clear_intraday_caches()
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
        f"{r['name']:<34} n={r['n']:.0f} days={r['n_days']:<3} "
        f"WR={r['wr']*100:5.1f}% PF={r['pf']:.2f} exp=${r['exp']:.2f} "
        f"pnl=${r['pnl']:.0f} dd={r['dd']*100:.2f}% gate={r['gate']}"
    )


def main() -> None:
    bars = pd.read_parquet(ROOT / "data" / "bars" / "SPY_5m.parquet")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("America/New_York")

    rows: list[dict] = []
    print("=== Intraday strategy research (ORB kept as baseline) ===")
    print(f"bars={len(bars)}  {bars.index.min()} -> {bars.index.max()}\n")

    rows.append(_run(bars, ["orb"], "0_baseline_orb"))
    print(_fmt(rows[-1]))

    print("\n--- Alone ---")
    for p in STRATEGIES:
        rows.append(_run(bars, [p], f"1_alone:{p}"))
        print(_fmt(rows[-1]))

    print("\n--- ORB + strategy ---")
    for p in STRATEGIES:
        rows.append(_run(bars, ["orb", p], f"2_orb+{p}"))
        print(_fmt(rows[-1]))

    # Also ORB + best candidates combo check: top alone by PF with n>=20
    print("\n--- Ranked by PF (n>=15) ---")
    eligible = [r for r in rows if r["n"] >= 15]
    for r in sorted(eligible, key=lambda x: (x["pf"], x["exp"]), reverse=True)[:12]:
        print(_fmt(r))

    out_csv = ROOT / "artifacts" / "intraday_strategies_research.csv"
    out_json = ROOT / "artifacts" / "intraday_strategies_research.json"
    pd.DataFrame([{k: v for k, v in r.items() if k != "by_pattern"} for r in rows]).to_csv(
        out_csv, index=False
    )
    out_json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
