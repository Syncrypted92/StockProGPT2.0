"""Compare AMD short-DTE exit styles: fixed TP vs swing / trail (research only).

Uses the locked live AMD detector (reclaim + HTF, 10:00–12:00, pierce 0.15).
Premium path is the same proxy as research_amd_short_dte_tune / backtest.
Does NOT change live config.
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

# Reuse signal precompute from short-DTE tune
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from research_amd_short_dte_tune import filt, precompute  # noqa: E402


def _pf(s: pd.Series) -> float:
    w = float(s[s > 0].sum())
    l = float((-s[s < 0]).sum())
    if l <= 1e-12:
        return float("inf") if w > 0 else 0.0
    return w / l


def _stats(pnls: list[float], holds: list[float] | None = None) -> dict:
    if not pnls:
        return {"n": 0, "wr": 0.0, "exp": 0.0, "pf": 0.0, "pnl": 0.0, "avg_hold_h": 0.0}
    s = pd.Series(pnls, dtype=float)
    out = {
        "n": int(len(s)),
        "wr": float((s > 0).mean()),
        "exp": float(s.mean()),
        "pf": _pf(s),
        "pnl": float(s.sum()),
        "p50": float(s.median()),
        "p90": float(s.quantile(0.9)),
        "worst": float(s.min()),
    }
    if holds:
        out["avg_hold_h"] = float(np.mean(holds))
    return out


def sim_exit(
    df: pd.DataFrame,
    sig: dict,
    *,
    mode: str,
    tp: float = 0.35,
    sl: float = 0.30,
    trail_pct: float = 0.20,
    arm_pct: float = 0.35,
    max_sessions: int = 3,
    gamma_scale: float = 0.55,
    overnight_theta: float = 0.03,
    premium_pct: float = 0.008,
) -> tuple[float, str, float]:
    """Return (pnl_$, reason, hold_hours)."""
    side = 1 if sig["side"] == "call" else -1
    spot = float(sig["spot"])
    entry = float(np.clip(spot * premium_pct, 0.80, 5.0)) * 1.02
    prem = entry
    sessions_seen = {sig["ts"].date()}
    last_date = sig["ts"].date()
    closes = df["Close"]
    i0 = int(sig["i"])
    peak_ret = 0.0
    armed = False
    reason = "eod"
    exit_j = i0

    for j in range(i0 + 1, len(df)):
        exit_j = j
        ts = df.index[j]
        d = ts.date()
        if d != last_date:
            prem *= 1.0 - overnight_theta
            sessions_seen.add(d)
            last_date = d
            if len(sessions_seen) > max_sessions:
                reason = "max_sessions"
                prem *= 0.98
                break

        prev, cur = float(closes.iloc[j - 1]), float(closes.iloc[j])
        if prev <= 0:
            continue
        und_ret = (cur / prev - 1.0) * side
        prem_frac = max(prem / spot, 1e-4)
        prem *= 1.0 + float(np.clip(gamma_scale * 0.40 * und_ret / prem_frac, -0.35, 0.80))
        ret = prem / entry - 1.0
        peak_ret = max(peak_ret, ret)

        # Hard stop always
        if ret <= -sl:
            reason = "stop_loss"
            prem *= 0.98
            break

        if mode == "fixed_tp":
            if ret >= tp:
                reason = "profit_target"
                prem *= 0.98
                break
        elif mode == "trail_after_arm":
            if ret >= arm_pct:
                armed = True
            if armed and ret <= peak_ret - trail_pct:
                reason = "trail_stop"
                prem *= 0.98
                break
            # Optional ceiling so runners don't become lotteries
            if ret >= 1.50:
                reason = "runner_cap"
                prem *= 0.98
                break
        elif mode == "hold_sessions":
            # Only SL + session time stop — no TP (pure swing)
            pass
        elif mode == "scale_half":
            # Approximate: bank half at arm, trail rest → effective exit ~ blend
            if ret >= arm_pct and not armed:
                armed = True
                # lock half of arm gain into floor via trailing from arm
                peak_ret = max(peak_ret, arm_pct)
            if armed and ret <= max(arm_pct * 0.5, peak_ret - trail_pct):
                reason = "scale_trail"
                prem *= 0.98
                break
            if ret >= 1.20:
                reason = "runner_cap"
                prem *= 0.98
                break
        else:
            raise ValueError(mode)

        if len(sessions_seen) >= max_sessions and ts.time() >= time(15, 45):
            reason = "time_stop"
            prem *= 0.98
            break

    hold_h = (df.index[exit_j] - sig["ts"]).total_seconds() / 3600.0
    pnl = (prem - entry) * 100
    return pnl, reason, hold_h


def main() -> None:
    bars = pd.read_parquet(ROOT / "data" / "bars" / "SPY_5m.parquet")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("America/New_York")
    df = enrich_bars(bars, orb_minutes=30)
    set_htf_cache(df)

    # Locked live recipe
    sigs = filt(
        precompute(df, confirm=False, htf=True, win_end=time(12, 0), pierce=0.15),
        gap_mode="none",
        min_atr_pct=None,
        skip_friday=False,
    )
    print(f"AMD locked recipe signals: n={len(sigs)}  {df.index.min().date()} -> {df.index.max().date()}\n")

    policies = [
        {"name": "live_fixed_tp35_sl30", "mode": "fixed_tp", "tp": 0.35, "sl": 0.30, "max_sessions": 3},
        {"name": "fixed_tp50_sl30", "mode": "fixed_tp", "tp": 0.50, "sl": 0.30, "max_sessions": 3},
        {"name": "fixed_tp70_sl30", "mode": "fixed_tp", "tp": 0.70, "sl": 0.30, "max_sessions": 3},
        {"name": "fixed_tp100_sl30", "mode": "fixed_tp", "tp": 1.00, "sl": 0.30, "max_sessions": 3},
        {"name": "trail_arm35_give20", "mode": "trail_after_arm", "arm_pct": 0.35, "trail_pct": 0.20, "sl": 0.30, "max_sessions": 3},
        {"name": "trail_arm35_give15", "mode": "trail_after_arm", "arm_pct": 0.35, "trail_pct": 0.15, "sl": 0.30, "max_sessions": 3},
        {"name": "trail_arm50_give25", "mode": "trail_after_arm", "arm_pct": 0.50, "trail_pct": 0.25, "sl": 0.30, "max_sessions": 3},
        {"name": "scale_half_arm35_trail20", "mode": "scale_half", "arm_pct": 0.35, "trail_pct": 0.20, "sl": 0.30, "max_sessions": 3},
        {"name": "hold_3sess_sl30_no_tp", "mode": "hold_sessions", "sl": 0.30, "max_sessions": 3},
        {"name": "hold_2sess_sl30_no_tp", "mode": "hold_sessions", "sl": 0.30, "max_sessions": 2},
        {"name": "trail_arm35_give20_ms2", "mode": "trail_after_arm", "arm_pct": 0.35, "trail_pct": 0.20, "sl": 0.30, "max_sessions": 2},
    ]

    rows = []
    reason_rows = []
    for pol in policies:
        name = pol["name"]
        kwargs = {k: v for k, v in pol.items() if k != "name"}
        pnls, holds, reasons = [], [], []
        for s in sigs:
            pnl, reason, hold_h = sim_exit(df, s, **kwargs)
            pnls.append(pnl)
            holds.append(hold_h)
            reasons.append(reason)
        st = _stats(pnls, holds)
        rc = pd.Series(reasons).value_counts(normalize=True).to_dict()
        row = {"policy": name, **kwargs, **st}
        rows.append(row)
        reason_rows.append({"policy": name, **{f"pct_{k}": float(v) for k, v in rc.items()}})
        print(
            f"{name:32s} n={st['n']:3d} WR={st['wr']*100:5.1f}% PF={st['pf']:5.2f} "
            f"exp=${st['exp']:6.2f} pnl=${st['pnl']:7.0f} "
            f"p90=${st['p90']:6.1f} worst=${st['worst']:6.1f} hold={st.get('avg_hold_h', 0):.1f}h"
        )

    clear_htf_cache()
    configure_amd_params({"require_confirm": False, "apply_htf": False, "window_end": time(12, 0)})
    configure_htf_permission(HtfPermissionConfig(enabled=False))

    out = ROOT / "artifacts" / "amd_swing_exit_research.csv"
    pd.DataFrame(rows).sort_values(["pf", "exp"], ascending=False).to_csv(out, index=False)
    pd.DataFrame(reason_rows).to_csv(ROOT / "artifacts" / "amd_swing_exit_reasons.csv", index=False)

    base = next(r for r in rows if r["policy"] == "live_fixed_tp35_sl30")
    best = max(rows, key=lambda r: (r["pf"] if r["pf"] == r["pf"] else 0, r["exp"]))
    print("\n=== vs live ===")
    print(
        f"live  PF={base['pf']:.2f} exp=${base['exp']:.2f} pnl=${base['pnl']:.0f} "
        f"WR={base['wr']*100:.1f}% hold={base['avg_hold_h']:.1f}h"
    )
    print(
        f"best  {best['policy']} PF={best['pf']:.2f} exp=${best['exp']:.2f} "
        f"pnl=${best['pnl']:.0f} WR={best['wr']*100:.1f}% hold={best['avg_hold_h']:.1f}h"
    )
    print(f"\nWrote {out}")
    print(
        "NOTE: option-premium PROXY only — same caveat as AMD WF. "
        "Today's +70% unmanaged hold is one path; incorporation needs this gate + paper trail."
    )


if __name__ == "__main__":
    main()
