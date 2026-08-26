"""Research open MTF liquidity sweeps (EQH/EQL on 1H/4H, entry on 5m)."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import time
from pathlib import Path

import pandas as pd

from stockpro.spy_day.backtest import run_spy_day_backtest
from stockpro.spy_day.mtf_liquidity import clear_htf_cache, set_htf_cache
from stockpro.spy_day.patterns import SWEEP_PARAMS
from stockpro.spy_day.session import SpyDayConfig, load_spy_day_config

ROOT = Path(__file__).resolve().parents[1]


VARIANTS: list[tuple[str, dict]] = [
    (
        "open_eq_htf_fade",
        {
            "window_start": time(9, 35),
            "window_end": time(10, 15),
            "eq_tol_pct": 0.0015,
            "max_level_dist_pct": 0.0035,
            "pierce_atr": 0.15,
            "min_touches": 2,
            "require_eq": True,
            "require_4h_level": False,
            "require_dual_tf": False,
            "bias_mode": "fade_with_htf",
            "min_vol_flood": 0.85,
            "allow_pdh_pdl": True,
            "one_per_day": True,
        },
    ),
    (
        "open_eq_strict_4h",
        {
            "window_start": time(9, 35),
            "window_end": time(10, 15),
            "eq_tol_pct": 0.0018,
            "max_level_dist_pct": 0.004,
            "pierce_atr": 0.12,
            "min_touches": 2,
            "require_eq": True,
            "require_4h_level": True,
            "require_dual_tf": False,
            "bias_mode": "fade_with_htf",
            "min_vol_flood": 0.90,
            "allow_pdh_pdl": False,
            "one_per_day": True,
        },
    ),
    (
        "open_dual_tf_eq",
        {
            "window_start": time(9, 35),
            "window_end": time(10, 20),
            "eq_tol_pct": 0.002,
            "max_level_dist_pct": 0.0045,
            "pierce_atr": 0.12,
            "min_touches": 2,
            "require_eq": True,
            "require_4h_level": False,
            "require_dual_tf": True,
            "bias_mode": "fade_with_htf",
            "min_vol_flood": 0.80,
            "allow_pdh_pdl": False,
            "one_per_day": True,
        },
    ),
    (
        "open_pdh_flood",
        {
            "window_start": time(9, 35),
            "window_end": time(10, 00),
            "eq_tol_pct": 0.0015,
            "max_level_dist_pct": 0.004,
            "pierce_atr": 0.18,
            "min_touches": 2,
            "require_eq": False,
            "require_4h_level": False,
            "require_dual_tf": False,
            "bias_mode": "none",
            "min_vol_flood": 1.0,
            "allow_pdh_pdl": True,
            "one_per_day": True,
        },
    ),
    (
        "open_eq_fade_only",
        {
            "window_start": time(9, 35),
            "window_end": time(10, 15),
            "eq_tol_pct": 0.0015,
            "max_level_dist_pct": 0.0035,
            "pierce_atr": 0.15,
            "min_touches": 2,
            "require_eq": True,
            "require_4h_level": False,
            "require_dual_tf": False,
            "bias_mode": "fade_only",
            "min_vol_flood": 0.85,
            "allow_pdh_pdl": True,
            "one_per_day": True,
        },
    ),
    (
        "open_wide_eq",
        {
            "window_start": time(9, 35),
            "window_end": time(10, 30),
            "eq_tol_pct": 0.0025,
            "max_level_dist_pct": 0.005,
            "pierce_atr": 0.10,
            "min_touches": 2,
            "require_eq": True,
            "require_4h_level": False,
            "require_dual_tf": False,
            "bias_mode": "fade_with_htf",
            "min_vol_flood": 0.70,
            "allow_pdh_pdl": True,
            "one_per_day": True,
        },
    ),
]


def _apply(params: dict) -> None:
    SWEEP_PARAMS.clear()
    SWEEP_PARAMS.update(params)


def _run(bars: pd.DataFrame, patterns: list[str], name: str) -> dict:
    base = load_spy_day_config()
    cfg = SpyDayConfig(
        **{
            **base.__dict__,
            "patterns": patterns,
            "score_threshold": 0.70,
            "pattern_min_confidence": {
                "orb": 0.70,
                "liquidity_sweep": 0.78,
            },
            "max_trades_per_day": 2,
        }
    )
    set_htf_cache(bars)
    try:
        res = run_spy_day_backtest(bars, cfg=cfg)
    finally:
        clear_htf_cache()
    m = res.metrics
    by = res.by_pattern.get("liquidity_sweep", {})
    return {
        "name": name,
        "patterns": patterns,
        "n": m.get("n_trades", 0),
        "wr": m.get("win_rate", 0),
        "exp": m.get("expectancy", 0),
        "pf": m.get("profit_factor", 0),
        "dd": m.get("max_drawdown", 0),
        "pnl": float(res.trades["pnl"].sum()) if len(res.trades) else 0.0,
        "sweep_n": by.get("n_trades", 0),
        "sweep_wr": by.get("win_rate", 0),
        "sweep_exp": by.get("expectancy", 0),
        "sweep_pf": by.get("profit_factor", 0),
        "sweep_pnl": by.get("total_pnl", 0),
        "gate_ok": bool(m.get("profit_factor", 0) >= base.backtest_gate_pf and m.get("n_trades", 0) >= base.backtest_gate_min_trades),
    }


def main() -> None:
    path = ROOT / "data" / "bars" / "SPY_5m.parquet"
    bars = pd.read_parquet(path)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("America/New_York")

    rows: list[dict] = []
    # Baseline ORB-only
    _apply(VARIANTS[0][1])
    rows.append(_run(bars, ["orb"], "baseline_orb_only"))

    defaults = deepcopy(SWEEP_PARAMS)
    for name, params in VARIANTS:
        _apply(params)
        rows.append(_run(bars, ["liquidity_sweep"], f"sweep_only:{name}"))
        rows.append(_run(bars, ["orb", "liquidity_sweep"], f"orb+sweep:{name}"))

    _apply(defaults)

    df = pd.DataFrame(rows)
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "open_sweep_research.csv"
    json_path = out_dir / "open_sweep_research.json"
    df.to_csv(csv_path, index=False)

    # Rank sweep-only by PF then expectancy (finite PF only)
    sweep_only = df[df["name"].str.startswith("sweep_only:")].copy()

    def _rank_pf(x: float) -> float:
        if x != x or x == float("inf"):
            return 0.0
        return float(x)

    sweep_only["pf_rank"] = sweep_only["pf"].map(_rank_pf)
    sweep_only = sweep_only.sort_values(["pf_rank", "exp", "n"], ascending=[False, False, False])

    combo = df[df["name"].str.startswith("orb+sweep:")].copy()
    combo["pf_rank"] = combo["pf"].map(_rank_pf)
    combo = combo.sort_values(["pf_rank", "exp"], ascending=[False, False])

    best_sweep = sweep_only.iloc[0].to_dict() if len(sweep_only) else {}
    best_combo = combo.iloc[0].to_dict() if len(combo) else {}
    baseline = df[df["name"] == "baseline_orb_only"].iloc[0].to_dict()

    payload = {
        "baseline_orb": baseline,
        "best_sweep_only": best_sweep,
        "best_orb_plus_sweep": best_combo,
        "all": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("=== Open MTF liquidity sweep research ===")
    print(f"Bars: {len(bars)}  {bars.index.min()} -> {bars.index.max()}")
    print("\nBaseline ORB-only:")
    print(
        f"  n={baseline['n']:.0f} WR={baseline['wr']:.1%} E=${baseline['exp']:.2f} "
        f"PF={baseline['pf']:.2f} gate={baseline['gate_ok']}"
    )
    print("\nSweep-only ranking:")
    for _, r in sweep_only.iterrows():
        print(
            f"  {r['name'][11:]:22s} n={r['n']:.0f} WR={r['wr']:.1%} "
            f"E=${r['exp']:.2f} PF={r['pf']:.2f} PnL=${r['pnl']:.0f}"
        )
    print("\nORB+sweep ranking:")
    for _, r in combo.iterrows():
        print(
            f"  {r['name'][10:]:22s} n={r['n']:.0f} WR={r['wr']:.1%} "
            f"E=${r['exp']:.2f} PF={r['pf']:.2f} sweep_n={r['sweep_n']:.0f} "
            f"sweep_PF={r['sweep_pf']:.2f} gate={r['gate_ok']}"
        )
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
