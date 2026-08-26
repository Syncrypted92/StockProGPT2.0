"""When is AMD swing favorable? Keep live TP35 default; find a gate for runners.

For each locked-recipe AMD signal, walk the premium proxy until first +35% (arm).
Record features at arm, then compare:
  A) exit at arm (live rule)
  B) trail after arm (give back 20%)
  C) hold to session max / max_sessions

Find simple gates where (B or C) beat A on the gated subset without killing the rest.
Research only — does not change live config.
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
from stockpro.spy_day.secondary_patterns import overnight_gap_pct

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from research_amd_short_dte_tune import filt, precompute  # noqa: E402

ARM = 0.35
SL = 0.30
TRAIL = 0.20
GAMMA = 0.55
THETA = 0.03
PREM_PCT = 0.008
MAX_SESS = 3


def _pf(s: pd.Series) -> float:
    w = float(s[s > 0].sum())
    l = float((-s[s < 0]).sum())
    if l <= 1e-12:
        return float("inf") if w > 0 else 0.0
    return w / l


def _stats(pnls: list[float]) -> dict:
    if not pnls:
        return {"n": 0, "wr": 0.0, "exp": 0.0, "pf": 0.0, "pnl": 0.0}
    s = pd.Series(pnls, dtype=float)
    return {
        "n": int(len(s)),
        "wr": float((s > 0).mean()),
        "exp": float(s.mean()),
        "pf": _pf(s),
        "pnl": float(s.sum()),
    }


def walk_trade(df: pd.DataFrame, sig: dict) -> dict | None:
    """Return arm features + pnl under several post-arm policies. None if never arms and not SL."""
    side = 1 if sig["side"] == "call" else -1
    spot0 = float(sig["spot"])
    entry = float(np.clip(spot0 * PREM_PCT, 0.80, 5.0)) * 1.02
    prem = entry
    sessions_seen = {sig["ts"].date()}
    last_date = sig["ts"].date()
    closes = df["Close"]
    highs = df["High"]
    lows = df["Low"]
    i0 = int(sig["i"])

    armed = False
    arm_j = None
    arm_prem = None
    peak_ret = 0.0
    stopped_before_arm = False

    # Path until arm or SL or max sessions
    for j in range(i0 + 1, len(df)):
        ts = df.index[j]
        d = ts.date()
        if d != last_date:
            prem *= 1.0 - THETA
            sessions_seen.add(d)
            last_date = d
            if len(sessions_seen) > MAX_SESS:
                break
        prev, cur = float(closes.iloc[j - 1]), float(closes.iloc[j])
        if prev <= 0:
            continue
        und_ret = (cur / prev - 1.0) * side
        prem_frac = max(prem / spot0, 1e-4)
        prem *= 1.0 + float(np.clip(GAMMA * 0.40 * und_ret / prem_frac, -0.35, 0.80))
        ret = prem / entry - 1.0
        if ret <= -SL:
            stopped_before_arm = True
            break
        if ret >= ARM:
            armed = True
            arm_j = j
            arm_prem = prem * 0.98  # same spread haircut as TP exit
            break
        if len(sessions_seen) >= MAX_SESS and ts.time() >= time(15, 45):
            break

    if not armed or arm_j is None or arm_prem is None:
        # Never reached TP — live and swing identical (SL / time)
        # Simulate full path with fixed SL only for baseline comparison rows
        pnl_live = (prem * 0.98 - entry) * 100 if stopped_before_arm else (prem * 0.98 - entry) * 100
        return {
            "armed": False,
            "stopped_before_arm": stopped_before_arm,
            "pnl_live": pnl_live,
            "pnl_trail": pnl_live,
            "pnl_hold": pnl_live,
            "swing_edge_trail": 0.0,
            "swing_edge_hold": 0.0,
        }

    # Features at arm
    ts_arm = df.index[arm_j]
    spot_arm = float(closes.iloc[arm_j])
    und_move = (spot_arm / spot0 - 1.0) * side  # favorable underlying move since entry
    hold_h = (ts_arm - sig["ts"]).total_seconds() / 3600.0
    sessions_at_arm = len({sig["ts"].date(), ts_arm.date()})
    # Same-day OR / day range context
    day_mask = df.index.date == sig["ts"].date()
    day = df.loc[day_mask]
    orb = day.iloc[:6] if len(day) >= 6 else day
    orb_hi = float(orb["High"].max()) if len(orb) else np.nan
    orb_lo = float(orb["Low"].min()) if len(orb) else np.nan
    day_hi = float(day["High"].max()) if len(day) else np.nan
    day_lo = float(day["Low"].min()) if len(day) else np.nan
    # Extension beyond OR in trade direction
    if side == 1:
        or_ext = (spot_arm - orb_hi) / spot0 if orb_hi == orb_hi else np.nan
        near_day_ext = (spot_arm - day_hi) / spot0 if day_hi == day_hi else np.nan
    else:
        or_ext = (orb_lo - spot_arm) / spot0 if orb_lo == orb_lo else np.nan
        near_day_ext = (day_lo - spot_arm) / spot0 if day_lo == day_lo else np.nan

    atr = float(df.iloc[arm_j]["ATR"]) if "ATR" in df.columns and pd.notna(df.iloc[arm_j].get("ATR")) else np.nan
    atr_pct = atr / spot_arm if atr == atr and spot_arm else np.nan
    gap = overnight_gap_pct(df, i0)
    gap_aligned = False
    if gap is not None:
        if side == 1 and gap < -0.0015:
            gap_aligned = True  # gap down then bull AMD
        if side == -1 and gap > 0.0015:
            gap_aligned = True

    # Momentum: last 6 bars (30m) in trade direction
    j0 = max(i0, arm_j - 6)
    mom = (float(closes.iloc[arm_j]) / float(closes.iloc[j0]) - 1.0) * side if j0 < arm_j else 0.0

    # After arm: trail and hold paths
    prem_t = arm_prem / 0.98  # continue from pre-haircut arm level then haircut on exit
    # Actually arm_prem already haircut; restart from pre-haircut for fair path
    prem_t = arm_prem / 0.98
    prem_h = prem_t
    peak_ret = ARM
    sessions_seen = {sig["ts"].date(), ts_arm.date()}
    last_date = ts_arm.date()
    trail_done = False
    trail_prem = None
    hold_prem = None

    for j in range(arm_j + 1, len(df)):
        ts = df.index[j]
        d = ts.date()
        if d != last_date:
            if not trail_done:
                prem_t *= 1.0 - THETA
            prem_h *= 1.0 - THETA
            sessions_seen.add(d)
            last_date = d
            if len(sessions_seen) > MAX_SESS:
                if not trail_done:
                    trail_prem = prem_t * 0.98
                    trail_done = True
                hold_prem = prem_h * 0.98
                break

        prev, cur = float(closes.iloc[j - 1]), float(closes.iloc[j])
        if prev <= 0:
            continue
        und_ret = (cur / prev - 1.0) * side
        for label, p_ref in (("t", prem_t), ("h", prem_h)):
            pass
        if not trail_done:
            prem_frac = max(prem_t / spot0, 1e-4)
            prem_t *= 1.0 + float(np.clip(GAMMA * 0.40 * und_ret / prem_frac, -0.35, 0.80))
            ret_t = prem_t / entry - 1.0
            peak_ret = max(peak_ret, ret_t)
            if ret_t <= -SL or ret_t <= peak_ret - TRAIL:
                trail_prem = prem_t * 0.98
                trail_done = True
            elif ret_t >= 1.50:
                trail_prem = prem_t * 0.98
                trail_done = True

        prem_frac_h = max(prem_h / spot0, 1e-4)
        prem_h *= 1.0 + float(np.clip(GAMMA * 0.40 * und_ret / prem_frac_h, -0.35, 0.80))
        ret_h = prem_h / entry - 1.0
        if ret_h <= -SL:
            hold_prem = prem_h * 0.98
            if not trail_done:
                trail_prem = prem_t * 0.98
                trail_done = True
            break
        if len(sessions_seen) >= MAX_SESS and ts.time() >= time(15, 45):
            hold_prem = prem_h * 0.98
            if not trail_done:
                trail_prem = prem_t * 0.98
                trail_done = True
            break

    if trail_prem is None:
        trail_prem = prem_t * 0.98
    if hold_prem is None:
        hold_prem = prem_h * 0.98

    pnl_live = (arm_prem - entry) * 100
    pnl_trail = (trail_prem - entry) * 100
    pnl_hold = (hold_prem - entry) * 100

    return {
        "armed": True,
        "ts_entry": str(sig["ts"]),
        "ts_arm": str(ts_arm),
        "side": sig["side"],
        "weekday": sig["weekday"],
        "hold_h_to_arm": hold_h,
        "sessions_at_arm": sessions_at_arm,
        "arm_hour_et": ts_arm.hour + ts_arm.minute / 60.0,
        "und_move_to_arm": und_move,
        "or_extension": or_ext,
        "day_ext": near_day_ext,
        "atr_pct": atr_pct,
        "mom_30m": mom,
        "gap": gap if gap is not None else np.nan,
        "gap_aligned": gap_aligned,
        "same_day_arm": sessions_at_arm == 1,
        "overnight_before_arm": sessions_at_arm >= 2,
        "pnl_live": pnl_live,
        "pnl_trail": pnl_trail,
        "pnl_hold": pnl_hold,
        "swing_edge_trail": pnl_trail - pnl_live,
        "swing_edge_hold": pnl_hold - pnl_live,
        "mfe_edge": max(pnl_trail, pnl_hold) - pnl_live,
    }


def eval_gate(armed_df: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    """Hybrid policy: trail when gate True else take live TP. Includes non-gated armed as live."""
    use_trail = mask.fillna(False)
    pnl = np.where(use_trail, armed_df["pnl_trail"], armed_df["pnl_live"])
    st = _stats(list(pnl))
    gated = armed_df[use_trail]
    ungated = armed_df[~use_trail]
    edge = float(gated["swing_edge_trail"].sum()) if len(gated) else 0.0
    live_all = _stats(list(armed_df["pnl_live"]))
    return {
        "gate": label,
        "n_gate": int(use_trail.sum()),
        "n_arm": int(len(armed_df)),
        "gate_wr_edge": float((gated["swing_edge_trail"] > 0).mean()) if len(gated) else 0.0,
        "gate_avg_edge": float(gated["swing_edge_trail"].mean()) if len(gated) else 0.0,
        "gate_edge_sum": edge,
        "hybrid_pf": st["pf"],
        "hybrid_exp": st["exp"],
        "hybrid_pnl": st["pnl"],
        "hybrid_wr": st["wr"],
        "live_pnl": live_all["pnl"],
        "delta_pnl": st["pnl"] - live_all["pnl"],
        "ungated_n": int(len(ungated)),
    }


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
    rows = [r for r in rows if r]
    all_df = pd.DataFrame(rows)
    armed = all_df[all_df["armed"]].copy()
    print(f"signals={len(sigs)}  armed(+35%)={len(armed)}  never_arm={int((~all_df['armed']).sum())}\n")

    if armed.empty:
        print("No armed trades — abort")
        return

    # Baseline: always live vs always trail on armed only
    print("=== Armed subset: take TP vs always trail ===")
    live = _stats(list(armed["pnl_live"]))
    trail = _stats(list(armed["pnl_trail"]))
    hold = _stats(list(armed["pnl_hold"]))
    print(f"take_tp35  n={live['n']} PF={live['pf']:.2f} exp=${live['exp']:.2f} pnl=${live['pnl']:.0f}")
    print(f"always_trail n={trail['n']} PF={trail['pf']:.2f} exp=${trail['exp']:.2f} pnl=${trail['pnl']:.0f}")
    print(f"always_hold  n={hold['n']} PF={hold['pf']:.2f} exp=${hold['exp']:.2f} pnl=${hold['pnl']:.0f}")
    print(
        f"trail beats TP on {(armed['swing_edge_trail'] > 0).mean()*100:.0f}% of armed; "
        f"avg edge=${armed['swing_edge_trail'].mean():.2f}\n"
    )

    gates = []
    # Candidate gates (simple, interpretable)
    candidates = [
        ("overnight_before_arm", armed["overnight_before_arm"] == True),  # noqa: E712
        ("same_day_arm", armed["same_day_arm"] == True),  # noqa: E712
        ("or_ext>0", armed["or_extension"] > 0),
        ("or_ext>0.001", armed["or_extension"] > 0.001),
        ("or_ext>0.002", armed["or_extension"] > 0.002),
        ("mom30>0", armed["mom_30m"] > 0),
        ("mom30>0.001", armed["mom_30m"] > 0.001),
        ("mom30>0.002", armed["mom_30m"] > 0.002),
        ("und_move>0.004", armed["und_move_to_arm"] > 0.004),
        ("und_move>0.006", armed["und_move_to_arm"] > 0.006),
        ("arm_after_14et", armed["arm_hour_et"] >= 14.0),
        ("arm_after_15et", armed["arm_hour_et"] >= 15.0),
        ("hold_to_arm>4h", armed["hold_h_to_arm"] > 4),
        ("hold_to_arm>20h", armed["hold_h_to_arm"] > 20),
        ("gap_aligned", armed["gap_aligned"] == True),  # noqa: E712
        ("call_only", armed["side"] == "call"),
        ("put_only", armed["side"] == "put"),
        # Combos inspired by today's path: overnight + extension + momentum
        (
            "overnight+or_ext>0+mom>0",
            (armed["overnight_before_arm"] == True)  # noqa: E712
            & (armed["or_extension"] > 0)
            & (armed["mom_30m"] > 0),
        ),
        (
            "overnight+mom>0.001",
            (armed["overnight_before_arm"] == True) & (armed["mom_30m"] > 0.001),  # noqa: E712
        ),
        (
            "or_ext>0.001+mom>0.001",
            (armed["or_extension"] > 0.001) & (armed["mom_30m"] > 0.001),
        ),
        (
            "or_ext>0+und>0.004+mom>0",
            (armed["or_extension"] > 0)
            & (armed["und_move_to_arm"] > 0.004)
            & (armed["mom_30m"] > 0),
        ),
        (
            "arm_late+or_ext>0",
            (armed["arm_hour_et"] >= 14.0) & (armed["or_extension"] > 0),
        ),
        (
            "overnight+or_ext>0",
            (armed["overnight_before_arm"] == True) & (armed["or_extension"] > 0),  # noqa: E712
        ),
    ]

    print("=== Hybrid: trail ONLY when gate true, else take TP35 ===")
    for label, mask in candidates:
        g = eval_gate(armed, mask, label)
        gates.append(g)

    gdf = pd.DataFrame(gates).sort_values(["delta_pnl", "gate_avg_edge"], ascending=False)
    # Prefer gates with enough samples and positive edge rate
    print(gdf.head(15).to_string(index=False))

    good = gdf[(gdf["delta_pnl"] > 0) & (gdf["n_gate"] >= 5) & (gdf["gate_avg_edge"] > 0)]
    print("\n=== Promising gates (delta_pnl>0, n_gate>=5, avg edge>0) ===")
    if good.empty:
        print("None — selective swing does not beat default on this proxy sample.")
    else:
        print(good.to_string(index=False))

    # Full-sample hybrid including never-armed (they contribute live path pnl)
    never = all_df[~all_df["armed"]]
    base_full = list(all_df["pnl_live"])
    print(f"\nFull sample always-TP (armed take TP; never-arm as resolved): {_stats(base_full)}")

    clear_htf_cache()
    configure_amd_params({"require_confirm": False, "apply_htf": False})
    configure_htf_permission(HtfPermissionConfig(enabled=False))

    out = ROOT / "artifacts" / "amd_swing_gate_research.csv"
    armed.to_csv(ROOT / "artifacts" / "amd_swing_arm_features.csv", index=False)
    gdf.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(f"Wrote {ROOT / 'artifacts' / 'amd_swing_arm_features.csv'}")


if __name__ == "__main__":
    main()
