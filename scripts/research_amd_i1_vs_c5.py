"""Deeper backtest: I1 (AMD call at +35%) vs C5 (call + morning + OR ext).

Default = hard TP at +35%. Indicator True -> trail 20% off peak (unless noted).
Locked AMD detector recipe. Proxy premiums (same as prior AMD research).
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
from research_amd_swing_gates import _stats, walk_trade  # noqa: E402


def _pf(s: pd.Series) -> float:
    w = float(s[s > 0].sum())
    l = float((-s[s < 0]).sum())
    if l <= 1e-12:
        return float("inf") if w > 0 else 0.0
    return w / l


def enrich(armed: pd.DataFrame) -> pd.DataFrame:
    d = armed.copy()
    d["I1"] = d["side"] == "call"
    d["C5"] = (
        (d["side"] == "call")
        & (d["arm_hour_et"] < 11.0)
        & (d["or_extension"] >= 0.0015)
    )
    d["I1_not_C5"] = d["I1"] & ~d["C5"]
    return d


def hybrid(armed: pd.DataFrame, mask: pd.Series, trail_col: str = "pnl_trail") -> list[float]:
    use = mask.fillna(False)
    return list(np.where(use, armed[trail_col], armed["pnl_live"]))


def resim_trail(
    df: pd.DataFrame,
    sigs_by_i: dict,
    armed_row: pd.Series,
    trail_pct: float,
    arm_pct: float = 0.35,
    sl: float = 0.30,
) -> float:
    """Re-walk one armed trade with alternate trail width; return pnl_$ (1 contract)."""
    # Find matching signal by ts
    ts = pd.Timestamp(armed_row["ts_entry"])
    # walk_trade already has pnl_trail at 20%; for sensitivity use research_amd_swing_exits style
    from research_amd_swing_exits import sim_exit

    # rebuild sig dict from row
    # need original sig - use ts match from precomputed list stored globally in main
    return float("nan")


def main() -> None:
    bars = pd.read_parquet(ROOT / "data" / "bars" / "SPY_5m.parquet")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("America/New_York")
    df = enrich_bars(bars, orb_minutes=30)
    set_htf_cache(df)

    raw_sigs = filt(
        precompute(df, confirm=False, htf=True, win_end=time(12, 0), pierce=0.15),
        gap_mode="none",
        min_atr_pct=None,
        skip_friday=False,
    )
    walked = [walk_trade(df, s) for s in raw_sigs]
    # attach entry index for trail sensitivity
    for s, w in zip(raw_sigs, walked):
        if w:
            w["sig"] = s
    all_df = pd.DataFrame([w for w in walked if w])
    never = all_df[~all_df["armed"]]
    armed = enrich(all_df[all_df["armed"]].copy())
    never_pnls = list(never["pnl_live"])

    print("=" * 72)
    print("I1 vs C5 - deeper AMD swing backtest")
    print("=" * 72)
    print(f"signals={len(raw_sigs)}  armed(+35%)={len(armed)}  never_arm={len(never)}")
    print(f"I1 fires={int(armed['I1'].sum())}  C5 fires={int(armed['C5'].sum())}  "
          f"I1_not_C5={int(armed['I1_not_C5'].sum())}")
    print()

    # --- Baselines ---
    policies = {
        "TP_always": pd.Series(False, index=armed.index),
        "I1_call": armed["I1"],
        "C5_morning_or": armed["C5"],
        "trail_always": pd.Series(True, index=armed.index),
    }

    print("--- Full sample (never-arm + hybrid on armed) ---")
    rows = []
    for name, mask in policies.items():
        pnls = never_pnls + hybrid(armed, mask)
        st = _stats(pnls)
        gated = armed[mask.fillna(False)]
        edge = float(gated["swing_edge_trail"].mean()) if len(gated) else 0.0
        edge_wr = float((gated["swing_edge_trail"] > 0).mean()) if len(gated) else 0.0
        row = {
            "policy": name,
            "n_fire": int(mask.sum()) if name != "TP_always" else 0,
            "edge_wr": edge_wr,
            "avg_edge": edge,
            **st,
            "delta_vs_tp": st["pnl"] - _stats(never_pnls + list(armed["pnl_live"]))["pnl"],
        }
        rows.append(row)
        print(
            f"{name:16s} fire={row['n_fire']:2d} edgeWR={edge_wr*100:5.1f}% "
            f"avgEdge=${edge:7.1f}  n={st['n']} WR={st['wr']*100:5.1f}% "
            f"PF={st['pf']:5.2f} exp=${st['exp']:6.2f} pnl=${st['pnl']:7.0f} "
            f"dTP=${row['delta_vs_tp']:7.0f}"
        )

    # --- Armed-only ---
    print("\n--- Armed subset only (hit +35%) ---")
    for name, mask in policies.items():
        st = _stats(hybrid(armed, mask))
        print(
            f"{name:16s} WR={st['wr']*100:5.1f}% PF={st['pf']:5.2f} "
            f"exp=${st['exp']:6.2f} pnl=${st['pnl']:7.0f}"
        )

    # --- Trade-level: I1 vs C5 overlap ---
    print("\n--- Trade-level armed calls ---")
    calls = armed[armed["I1"]].copy()
    calls["or_bps"] = calls["or_extension"] * 10000
    print(
        calls[
            ["ts_entry", "ts_arm", "arm_hour_et", "or_bps", "C5", "pnl_live", "pnl_trail", "swing_edge_trail"]
        ].to_string(index=False)
    )
    only_i1 = armed[armed["I1_not_C5"]]
    print(f"\nI1 but NOT C5 (n={len(only_i1)}): avg edge vs TP ${only_i1['swing_edge_trail'].mean():.1f}")
    if len(only_i1):
        print(only_i1[["ts_entry", "arm_hour_et", "or_extension", "pnl_live", "pnl_trail", "swing_edge_trail"]].to_string(index=False))

    # --- Expanding walk-forward on armed chronologically ---
    print("\n--- Expanding walk-forward (armed only, min 5 IS) ---")
    armed_s = armed.sort_values("ts_entry").reset_index(drop=True)
    wf_rows = []
    for k in range(5, len(armed_s)):
        is_df = armed_s.iloc[:k]
        oos = armed_s.iloc[k : k + 1]
        for pol, col in [("I1", "I1"), ("C5", "C5")]:
            # pick policy fixed (not re-fit) — just OOS one-step
            mask_oos = oos[col]
            h = hybrid(oos, mask_oos)[0]
            live = float(oos["pnl_live"].iloc[0])
            wf_rows.append(
                {
                    "k_is": k,
                    "oos_ts": str(oos["ts_entry"].iloc[0]),
                    "policy": pol,
                    "fired": bool(mask_oos.iloc[0]),
                    "oos_pnl": h,
                    "live_pnl": live,
                    "delta": h - live,
                }
            )
    wf = pd.DataFrame(wf_rows)
    for pol in ["I1", "C5"]:
        sub = wf[wf["policy"] == pol]
        print(
            f"{pol}: OOS steps={len(sub)} fired={int(sub['fired'].sum())} "
            f"sum_delta_vs_TP=${sub['delta'].sum():.0f} "
            f"mean_delta=${sub['delta'].mean():.1f} "
            f"pct_delta>0={((sub['delta']>0).mean()*100):.0f}%"
        )

    # --- 60/40 holdout ---
    print("\n--- 60/40 time holdout (armed) ---")
    cut = max(3, int(len(armed_s) * 0.6))
    for split, part in [("IS", armed_s.iloc[:cut]), ("OOS", armed_s.iloc[cut:])]:
        live = _stats(list(part["pnl_live"]))
        i1 = _stats(hybrid(part, part["I1"]))
        c5 = _stats(hybrid(part, part["C5"]))
        print(
            f"{split:3s} n={len(part)} live_pnl=${live['pnl']:.0f} | "
            f"I1 pnl=${i1['pnl']:.0f} (d={i1['pnl']-live['pnl']:.0f}) | "
            f"C5 pnl=${c5['pnl']:.0f} (d={c5['pnl']-live['pnl']:.0f})"
        )

    # --- Trail sensitivity on I1 vs C5 (re-sim) ---
    print("\n--- Trail width sensitivity (full sample dPnL vs TP) ---")
    from research_amd_swing_exits import sim_exit

    base_tp_full = _stats(never_pnls + list(armed["pnl_live"]))["pnl"]
    sens_rows = []
    for trail in (0.10, 0.15, 0.20, 0.25, 0.30):
        # map armed rows back to sigs via ts_entry
        trail_pnls_by_ts = {}
        for s in raw_sigs:
            pnl, reason, _hold = sim_exit(
                df,
                s,
                mode="trail_after_arm",
                arm_pct=0.35,
                trail_pct=trail,
                sl=0.30,
                max_sessions=3,
            )
            # only matters for those that arm; walk_trade marks armed
            trail_pnls_by_ts[str(s["ts"])] = pnl
        # For armed trades use this trail pnl; unarmed use live from walk
        # Rebuild armed trail column
        a2 = armed.copy()
        a2["pnl_trail_alt"] = [
            trail_pnls_by_ts.get(str(pd.Timestamp(ts)), live)
            for ts, live in zip(a2["ts_entry"], a2["pnl_live"])
        ]
        # For trades that never armed in sim_exit trail mode, pnl is full path —
        # align: only replace when I1/C5 would use trail. Use swing edge = alt - live for armed that hit arm in walk_trade.
        for name, mask in [("I1", a2["I1"]), ("C5", a2["C5"])]:
            pnls = never_pnls + list(np.where(mask, a2["pnl_trail_alt"], a2["pnl_live"]))
            st = _stats(pnls)
            sens_rows.append(
                {
                    "trail_pct": trail,
                    "policy": name,
                    "full_pnl": st["pnl"],
                    "full_pf": st["pf"],
                    "delta_vs_tp": st["pnl"] - base_tp_full,
                }
            )
            print(
                f"trail={trail:.0%} {name:3s}  full_pnl=${st['pnl']:7.0f} PF={st['pf']:.2f} "
                f"dTP=${st['pnl']-base_tp_full:7.0f}"
            )

    # --- Bootstrap armed deltas (I1 / C5) ---
    print("\n--- Bootstrap mean edge vs TP on fires (5000x) ---")
    rng = np.random.default_rng(42)
    boot_rows = []
    for name, mask in [("I1", armed["I1"]), ("C5", armed["C5"])]:
        edges = armed.loc[mask, "swing_edge_trail"].to_numpy()
        if len(edges) == 0:
            continue
        means = []
        for _ in range(5000):
            sample = rng.choice(edges, size=len(edges), replace=True)
            means.append(sample.mean())
        means = np.array(means)
        boot_rows.append(
            {
                "policy": name,
                "n": len(edges),
                "mean_edge": float(edges.mean()),
                "p05": float(np.quantile(means, 0.05)),
                "p50": float(np.quantile(means, 0.50)),
                "p95": float(np.quantile(means, 0.95)),
                "pct_mean_gt_0": float((means > 0).mean()),
            }
        )
        print(
            f"{name}: n={len(edges)} mean_edge=${edges.mean():.1f} "
            f"boot 90% CI [${np.quantile(means,0.05):.1f}, ${np.quantile(means,0.95):.1f}] "
            f"P(mean>0)={(means>0).mean()*100:.1f}%"
        )

    clear_htf_cache()
    configure_amd_params({"require_confirm": False, "apply_htf": False})
    configure_htf_permission(HtfPermissionConfig(enabled=False))

    out = ROOT / "artifacts" / "amd_i1_vs_c5_backtest.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(sens_rows).to_csv(ROOT / "artifacts" / "amd_i1_vs_c5_trail_sens.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(ROOT / "artifacts" / "amd_i1_vs_c5_bootstrap.csv", index=False)
    wf.to_csv(ROOT / "artifacts" / "amd_i1_vs_c5_walkforward.csv", index=False)
    armed.to_csv(ROOT / "artifacts" / "amd_i1_vs_c5_armed_trades.csv", index=False)
    print(f"\nWrote {out}")
    print("Caveat: proxy premiums; I1 n=5 / C5 n=3 armed calls — fragile.")


if __name__ == "__main__":
    main()
