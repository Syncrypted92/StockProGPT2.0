"""Clear AMD swing INDICATORS + backtest: when to trail vs take TP35.

Default rule stays: take profit at +35%.
Each indicator is a yes/no flag evaluated AT the moment premium first hits +35%.
Hybrid policy: if indicator True → trail 20% off peak; else → bank TP.

Proxy premium path = locked AMD recipe (same as prior AMD research).
Research only — does not change live config by itself.
"""

from __future__ import annotations

from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from stockpro.spy_day.amd import configure_amd_params
from stockpro.spy_day.htf_permission import HtfPermissionConfig
from stockpro.spy_day.mtf_liquidity import clear_htf_cache, set_htf_cache
from stockpro.spy_day.patterns import configure_htf_permission, enrich_bars

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from research_amd_short_dte_tune import filt, precompute  # noqa: E402
from research_amd_swing_gates import walk_trade, _stats  # noqa: E402

# Indicator catalog — clear definitions for humans
INDICATORS: dict[str, str] = {
    "I1_call": "Option is a CALL (not put)",
    "I2_same_day_arm": "Hit +35% same session (no overnight yet)",
    "I3_arm_before_11": "Armed before 11:00 ET (morning continuation)",
    "I4_arm_before_14": "Armed before 14:00 ET (not late-day)",
    "I5_strong_und_1pct": "Underlying moved >=1.0% in trade direction by arm",
    "I6_strong_und_1p5": "Underlying moved >=1.5% in trade direction by arm",
    "I7_or_ext_15bps": "Price >=15 bps beyond opening range in trade direction",
    "I8_or_ext_40bps": "Price >=40 bps beyond opening range in trade direction",
    "I9_mom30_pos": "Last 30m momentum still in trade direction at arm",
    "I10_mom30_gt_20bps": "Last 30m momentum >=20 bps in trade direction",
    "I11_gap_aligned": "Overnight gap opposed entry (gap-down->call / gap-up->put)",
    "I12_hold_to_arm_lt_6h": "Reached +35% within 6 hours of entry",
    # Combos (still readable)
    "C1_call_and_mom": "CALL + 30m momentum still with trade",
    "C2_call_and_or_ext": "CALL + >=15 bps beyond OR",
    "C3_call_and_strong_und": "CALL + underlying >=1.0% favorable",
    "C4_call_fast_arm": "CALL + same-day arm (hit TP without overnight)",
    "C5_call_morning_ext": "CALL + arm before 11 ET + beyond OR",
    "C6_put_never": "Always False (sanity: never swing) - equals live TP",
    "C7_always_swing": "Always True (sanity: always trail) - worst case",
}


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["I1_call"] = d["side"] == "call"
    d["I2_same_day_arm"] = d["same_day_arm"] == True  # noqa: E712
    d["I3_arm_before_11"] = d["arm_hour_et"] < 11.0
    d["I4_arm_before_14"] = d["arm_hour_et"] < 14.0
    d["I5_strong_und_1pct"] = d["und_move_to_arm"] >= 0.010
    d["I6_strong_und_1p5"] = d["und_move_to_arm"] >= 0.015
    d["I7_or_ext_15bps"] = d["or_extension"] >= 0.0015
    d["I8_or_ext_40bps"] = d["or_extension"] >= 0.0040
    d["I9_mom30_pos"] = d["mom_30m"] > 0
    d["I10_mom30_gt_20bps"] = d["mom_30m"] >= 0.002
    d["I11_gap_aligned"] = d["gap_aligned"] == True  # noqa: E712
    d["I12_hold_to_arm_lt_6h"] = d["hold_h_to_arm"] < 6.0
    d["C1_call_and_mom"] = d["I1_call"] & d["I9_mom30_pos"]
    d["C2_call_and_or_ext"] = d["I1_call"] & d["I7_or_ext_15bps"]
    d["C3_call_and_strong_und"] = d["I1_call"] & d["I5_strong_und_1pct"]
    d["C4_call_fast_arm"] = d["I1_call"] & d["I2_same_day_arm"]
    d["C5_call_morning_ext"] = d["I1_call"] & d["I3_arm_before_11"] & d["I7_or_ext_15bps"]
    d["C6_put_never"] = False
    d["C7_always_swing"] = True
    return d


def hybrid_pnl(armed: pd.DataFrame, mask: pd.Series) -> list[float]:
    use = mask.fillna(False)
    return list(np.where(use, armed["pnl_trail"], armed["pnl_live"]))


def main() -> None:
    bars = pd.read_parquet(ROOT / "data" / "bars" / "SPY_5m.parquet")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("America/New_York")
    df = enrich_bars(bars, orb_minutes=30)
    set_htf_cache(df)

    sigs = filt(
        precompute(df, confirm=False, htf=True, win_end=time(12, 0), pierce=0.15),
        gap_mode="none",
        min_atr_pct=None,
        skip_friday=False,
    )
    rows = [walk_trade(df, s) for s in sigs]
    all_df = pd.DataFrame([r for r in rows if r])
    never = all_df[~all_df["armed"]].copy()
    armed = add_indicators(all_df[all_df["armed"]].copy())

    print("=" * 72)
    print("AMD SWING INDICATORS - backtest (locked recipe)")
    print("=" * 72)
    print(f"signals={len(sigs)}  hit_+35%={len(armed)}  never_arm={len(never)}")
    print("Default: bank TP at +35%. Indicator True -> trail 20% instead.\n")

    print("--- Indicator dictionary ---")
    for k, desc in INDICATORS.items():
        print(f"  {k:24s}  {desc}")

    # Baseline on armed + full sample
    live_armed = _stats(list(armed["pnl_live"]))
    trail_armed = _stats(list(armed["pnl_trail"]))
    never_pnls = list(never["pnl_live"])
    full_live = _stats(never_pnls + list(armed["pnl_live"]))
    print("\n--- Baselines ---")
    print(
        f"armed take_TP35   n={live_armed['n']} WR={live_armed['wr']*100:.0f}% "
        f"PF={live_armed['pf']:.2f} exp=${live_armed['exp']:.1f} pnl=${live_armed['pnl']:.0f}"
    )
    print(
        f"armed always_trail n={trail_armed['n']} WR={trail_armed['wr']*100:.0f}% "
        f"PF={trail_armed['pf']:.2f} exp=${trail_armed['exp']:.1f} pnl=${trail_armed['pnl']:.0f}"
    )
    print(
        f"FULL sample TP35  n={full_live['n']} WR={full_live['wr']*100:.0f}% "
        f"PF={full_live['pf']:.2f} exp=${full_live['exp']:.1f} pnl=${full_live['pnl']:.0f}"
    )

    results = []
    print("\n--- Backtest each indicator (armed subset + full-sample hybrid) ---")
    hdr = (
        f"{'indicator':24s} {'n_fire':>6} {'edge%':>6} {'avgEdge':>8} "
        f"{'armPF':>6} {'arm$':>7} {'fullPF':>6} {'full$':>7} {'dFull$':>7}  definition"
    )
    print(hdr)
    print("-" * len(hdr))

    for key, desc in INDICATORS.items():
        mask = armed[key] if key in armed.columns else pd.Series(False, index=armed.index)
        n_fire = int(mask.sum())
        gated = armed[mask]
        edge_rate = float((gated["swing_edge_trail"] > 0).mean()) if n_fire else 0.0
        avg_edge = float(gated["swing_edge_trail"].mean()) if n_fire else 0.0
        arm_h = _stats(hybrid_pnl(armed, mask))
        full_h = _stats(never_pnls + hybrid_pnl(armed, mask))
        d_full = full_h["pnl"] - full_live["pnl"]
        results.append(
            {
                "indicator": key,
                "definition": desc,
                "n_fire": n_fire,
                "n_arm": len(armed),
                "edge_win_rate": edge_rate,
                "avg_edge_vs_tp": avg_edge,
                "armed_hybrid_pf": arm_h["pf"],
                "armed_hybrid_pnl": arm_h["pnl"],
                "armed_hybrid_exp": arm_h["exp"],
                "full_hybrid_pf": full_h["pf"],
                "full_hybrid_pnl": full_h["pnl"],
                "full_hybrid_exp": full_h["exp"],
                "delta_full_pnl_vs_tp": d_full,
                "beats_tp": bool(d_full > 1.0 and n_fire >= 3 and avg_edge > 0),
            }
        )
        flag = " <--" if results[-1]["beats_tp"] else ""
        print(
            f"{key:24s} {n_fire:6d} {edge_rate*100:5.0f}% {avg_edge:8.1f} "
            f"{arm_h['pf']:6.2f} {arm_h['pnl']:7.0f} {full_h['pf']:6.2f} "
            f"{full_h['pnl']:7.0f} {d_full:7.0f}  {desc[:42]}{flag}"
        )

    res = pd.DataFrame(results).sort_values(
        ["beats_tp", "delta_full_pnl_vs_tp", "avg_edge_vs_tp"],
        ascending=[False, False, False],
    )

    # Simple time holdout on armed trades (first 60% / last 40% by entry time)
    armed_sorted = armed.sort_values("ts_entry")
    cut = max(3, int(len(armed_sorted) * 0.6))
    is_oos = pd.Series(False, index=armed_sorted.index)
    is_oos.iloc[cut:] = True
    print("\n--- Holdout on armed trades (last ~40% by time) ---")
    print(f"IS n={cut}  OOS n={len(armed_sorted)-cut}")
    holdout_rows = []
    for key in ["I1_call", "C1_call_and_mom", "C2_call_and_or_ext", "C3_call_and_strong_und", "C4_call_fast_arm", "C5_call_morning_ext"]:
        mask = armed_sorted[key]
        for split, m in [("IS", ~is_oos), ("OOS", is_oos)]:
            sub = armed_sorted[m]
            smask = mask[m]
            if len(sub) == 0:
                continue
            h = _stats(hybrid_pnl(sub, smask))
            live = _stats(list(sub["pnl_live"]))
            holdout_rows.append(
                {
                    "indicator": key,
                    "split": split,
                    "n": len(sub),
                    "n_fire": int(smask.sum()),
                    "hybrid_pnl": h["pnl"],
                    "live_pnl": live["pnl"],
                    "delta": h["pnl"] - live["pnl"],
                }
            )
    hdf = pd.DataFrame(holdout_rows)
    print(hdf.to_string(index=False))

    winners = res[res["beats_tp"]]
    print("\n=== CLEAR SIGNALS THAT BEAT HARD TP (full sample) ===")
    if winners.empty:
        print("None with n_fire>=3 and positive avg edge.")
    else:
        for _, r in winners.iterrows():
            print(
                f"  {r['indicator']}: {r['definition']}\n"
                f"    fires={r['n_fire']}/{r['n_arm']} armed | "
                f"trail-beats-TP {r['edge_win_rate']*100:.0f}% of fires | "
                f"avg edge ${r['avg_edge_vs_tp']:.0f} | "
                f"full dPnL ${r['delta_full_pnl_vs_tp']:.0f}"
            )

    clear_htf_cache()
    configure_amd_params({"require_confirm": False, "apply_htf": False})
    configure_htf_permission(HtfPermissionConfig(enabled=False))

    out = ROOT / "artifacts" / "amd_swing_indicator_backtest.csv"
    res.to_csv(out, index=False)
    armed.to_csv(ROOT / "artifacts" / "amd_swing_arm_with_indicators.csv", index=False)
    hdf.to_csv(ROOT / "artifacts" / "amd_swing_indicator_holdout.csv", index=False)
    print(f"\nWrote {out}")
    print("NOTE: proxy premiums; armed n=18 is thin - treat as paper hypothesis, not proof.")


if __name__ == "__main__":
    main()
