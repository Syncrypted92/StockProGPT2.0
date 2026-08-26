"""Methodical research: secondary patterns alone and paired with ORB.

Strategies
  1. orb_gap_align
  2. orb_retest
  3. vwap_orb_bias
  4. pdh_pdl_killzone
  5. ib_failure

Baseline: live ORB + soft HTF (eq_context=none).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stockpro.spy_day.backtest import run_spy_day_backtest
from stockpro.spy_day.htf_permission import HtfPermissionConfig
from stockpro.spy_day.session import SpyDayConfig, load_spy_day_config

ROOT = Path(__file__).resolve().parents[1]

SECONDARIES = [
    "orb_gap_align",
    "orb_retest",
    "vwap_orb_bias",
    "pdh_pdl_killzone",
    "ib_failure",
]


def _cfg(patterns: list[str], *, soft_htf: bool = True) -> SpyDayConfig:
    base = load_spy_day_config()
    htf = HtfPermissionConfig(
        enabled=True,
        skip_4h_counter_trend=True,
        eq_context="none" if soft_htf else "leave",
        eq_tol_pct=0.0025,
        eq_max_dist_pct=0.004,
        eq_min_touches=2,
    )
    # orb_gap_align already embeds ORB+HTF via _orb_signal
    pmc = {
        "orb": 0.70,
        "orb_gap_align": 0.70,
        "orb_retest": 0.74,
        "vwap_orb_bias": 0.72,
        "pdh_pdl_killzone": 0.73,
        "ib_failure": 0.74,
    }
    return SpyDayConfig(
        **{
            **{k: v for k, v in base.__dict__.items() if k not in ("htf_permission", "patterns", "pattern_min_confidence")},
            "patterns": patterns,
            "score_threshold": 0.70,
            "pattern_min_confidence": pmc,
            "max_trades_per_day": 2,
            "htf_permission": htf,
        }
    )


def _run(bars: pd.DataFrame, patterns: list[str], name: str) -> dict:
    res = run_spy_day_backtest(bars, cfg=_cfg(patterns))
    m = res.metrics
    by = {
        k: {kk: float(vv) for kk, vv in d.items()}
        for k, d in res.by_pattern.items()
    }
    n_days = int(res.trades["date"].nunique()) if len(res.trades) and "date" in res.trades.columns else 0
    return {
        "name": name,
        "patterns": patterns,
        "n": float(m.get("n_trades", 0)),
        "n_days": n_days,
        "tpd": float(m.get("trades_per_day", 0)),
        "wr": float(m.get("win_rate", 0)),
        "exp": float(m.get("expectancy", 0)),
        "pf": float(m.get("profit_factor", 0) if m.get("profit_factor", 0) == m.get("profit_factor", 0) else 0),
        "pnl": float(res.trades["pnl"].sum()) if len(res.trades) else 0.0,
        "dd": float(m.get("max_drawdown", 0)),
        "gate": bool(
            m.get("profit_factor", 0) >= 1.2
            and m.get("n_trades", 0) >= 40
            and m.get("profit_factor", 0) == m.get("profit_factor", 0)
        ),
        "by_pattern": by,
    }


def main() -> None:
    bars = pd.read_parquet(ROOT / "data" / "bars" / "SPY_5m.parquet")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("America/New_York")

    rows: list[dict] = []
    print("=== Secondary pattern research (methodical) ===")
    print(f"bars={len(bars)}  {bars.index.min()} -> {bars.index.max()}\n")

    # 0) Baseline
    rows.append(_run(bars, ["orb"], "0_baseline_orb"))
    print(_fmt(rows[-1]))

    # 1) Each alone
    print("\n--- Alone ---")
    for p in SECONDARIES:
        rows.append(_run(bars, [p], f"1_alone:{p}"))
        print(_fmt(rows[-1]))

    # 2) ORB + each (ORB priority kept via PATTERN_PRIORITY)
    print("\n--- ORB + secondary ---")
    for p in SECONDARIES:
        rows.append(_run(bars, ["orb", p], f"2_orb+{p}"))
        print(_fmt(rows[-1]))

    # Rank by PF then expectancy among gated or near-gated
    ranked = sorted(rows, key=lambda r: (r["pf"], r["exp"], r["n"]), reverse=True)
    print("\n=== Ranked by PF ===")
    for r in ranked:
        print(_fmt(r))

    out = ROOT / "artifacts" / "secondary_patterns_research.json"
    out.write_text(json.dumps({"variants": rows, "ranked": ranked}, indent=2), encoding="utf-8")
    csv = ROOT / "artifacts" / "secondary_patterns_research.csv"
    pd.DataFrame([{k: v for k, v in r.items() if k != "by_pattern"} for r in rows]).to_csv(csv, index=False)
    print(f"\nwrote {out}")
    print(f"wrote {csv}")


def _fmt(r: dict) -> str:
    return (
        f"{r['name'][:36]:36s} n={r['n']:5.0f} tpd={r['tpd']:.2f} "
        f"WR={r['wr']:5.1%} E=${r['exp']:7.2f} PF={r['pf']:5.2f} "
        f"PnL=${r['pnl']:7.0f} gate={r['gate']}"
    )


if __name__ == "__main__":
    main()
