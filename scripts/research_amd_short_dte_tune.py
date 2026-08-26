"""Tune AMD short-dated (~3 DTE) proxy — fast grid.

Precompute AMD signals for detector knobs, then post-filter + exit economics.
"""

from __future__ import annotations

import itertools
import json
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from stockpro.spy_day.amd import AMD_PARAMS, amd_signal, configure_amd_params
from stockpro.spy_day.htf_permission import HtfPermissionConfig
from stockpro.spy_day.mtf_liquidity import clear_htf_cache, set_htf_cache
from stockpro.spy_day.patterns import configure_htf_permission, enrich_bars
from stockpro.spy_day.secondary_patterns import overnight_gap_pct

ROOT = Path(__file__).resolve().parents[1]


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


def precompute(df: pd.DataFrame, *, confirm: bool, htf: bool, win_end: time, pierce: float) -> list[dict]:
    configure_amd_params(
        {
            "window_start": time(10, 0),
            "window_end": win_end,
            "pierce_atr": pierce,
            "one_per_day": True,
            "require_confirm": confirm,
            "apply_htf": htf,
            "min_body_frac": 0.35,
        }
    )
    AMD_PARAMS["window_end"] = win_end
    configure_htf_permission(
        HtfPermissionConfig(enabled=True, skip_4h_counter_trend=True, eq_context="none")
        if htf
        else HtfPermissionConfig(enabled=False)
    )
    out: list[dict] = []
    for i in range(len(df)):
        sig = amd_signal(df, i)
        if not sig:
            continue
        d = df.index[i].date()
        if out and out[-1]["ts"].date() == d:
            continue
        atr = float(df.iloc[i]["ATR"]) if pd.notna(df.iloc[i].get("ATR")) else np.nan
        gap = overnight_gap_pct(df, i)
        out.append(
            {
                "i": i,
                "ts": df.index[i],
                "side": sig.side,
                "spot": float(sig.spot),
                "atr": atr,
                "atr_pct": (atr / float(sig.spot)) if atr == atr and sig.spot else np.nan,
                "gap": gap,
                "weekday": df.index[i].weekday(),
            }
        )
    return out


def filt(sigs: list[dict], *, gap_mode: str, min_atr_pct: float | None, skip_friday: bool) -> list[dict]:
    out = []
    for s in sigs:
        if skip_friday and s["weekday"] == 4:
            continue
        if min_atr_pct is not None:
            ap = s["atr_pct"]
            if ap != ap or ap < min_atr_pct:
                continue
        gap = s["gap"]
        if gap_mode == "with_fill":
            if gap is None or abs(gap) < 0.0015:
                continue
            if gap > 0 and s["side"] != "put":
                continue
            if gap < 0 and s["side"] != "call":
                continue
        elif gap_mode == "with_go":
            if gap is None or abs(gap) < 0.0015:
                continue
            if gap > 0 and s["side"] != "call":
                continue
            if gap < 0 and s["side"] != "put":
                continue
        out.append(s)
    return out


def sim_short_dte(
    df: pd.DataFrame,
    sig: dict,
    *,
    max_sessions: int,
    tp: float,
    sl: float,
    gamma_scale: float,
    overnight_theta: float,
    premium_pct: float,
) -> float:
    side = 1 if sig["side"] == "call" else -1
    spot = float(sig["spot"])
    entry = float(np.clip(spot * premium_pct, 0.80, 5.0)) * 1.02
    prem = entry
    sessions_seen = {sig["ts"].date()}
    last_date = sig["ts"].date()
    closes = df["Close"]
    i0 = int(sig["i"])
    for j in range(i0 + 1, len(df)):
        ts = df.index[j]
        d = ts.date()
        if d != last_date:
            prem *= 1.0 - overnight_theta
            sessions_seen.add(d)
            last_date = d
            if len(sessions_seen) > max_sessions:
                break
        prev, cur = float(closes.iloc[j - 1]), float(closes.iloc[j])
        if prev <= 0:
            continue
        und_ret = (cur / prev - 1.0) * side
        prem_frac = max(prem / spot, 1e-4)
        prem *= 1.0 + float(np.clip(gamma_scale * 0.40 * und_ret / prem_frac, -0.35, 0.80))
        ret = prem / entry - 1.0
        if ret >= tp:
            prem *= 0.98
            break
        if ret <= -sl:
            prem *= 0.98
            break
        if len(sessions_seen) >= max_sessions and ts.time() >= time(15, 45):
            prem *= 0.98
            break
    return (prem - entry) * 100


def main() -> None:
    bars = pd.read_parquet(ROOT / "data" / "bars" / "SPY_5m.parquet")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("America/New_York")
    df = enrich_bars(bars, orb_minutes=30)
    set_htf_cache(df)

    print("=== AMD short-DTE tune (fast) ===")
    print(f"bars={len(df)}  {df.index.min()} -> {df.index.max()}\n")

    detector_keys = list(
        itertools.product(
            [False, True],  # confirm
            [True],  # htf
            [time(11, 30), time(12, 0), time(13, 0)],
            [0.12, 0.15, 0.22],
        )
    )
    cache: dict[tuple, list[dict]] = {}
    print(f"Precompute {len(detector_keys)} detector combos...", flush=True)
    for n, (confirm, htf, win_end, pierce) in enumerate(detector_keys, 1):
        key = (confirm, htf, win_end, pierce)
        cache[key] = precompute(df, confirm=confirm, htf=htf, win_end=win_end, pierce=pierce)
        print(f"  [{n}/{len(detector_keys)}] confirm={confirm} end={win_end} pierce={pierce} -> {len(cache[key])} sigs", flush=True)

    filters = list(
        itertools.product(
            ["none", "with_fill"],
            [None, 0.00055],
            [False, True],  # skip friday
        )
    )
    exits = [
        {"max_sessions": 2, "tp": 0.30, "sl": 0.25, "gamma_scale": 0.55, "overnight_theta": 0.03, "premium_pct": 0.008},
        {"max_sessions": 3, "tp": 0.35, "sl": 0.30, "gamma_scale": 0.55, "overnight_theta": 0.03, "premium_pct": 0.008},
        {"max_sessions": 3, "tp": 0.40, "sl": 0.25, "gamma_scale": 0.55, "overnight_theta": 0.03, "premium_pct": 0.008},
        {"max_sessions": 3, "tp": 0.35, "sl": 0.30, "gamma_scale": 0.45, "overnight_theta": 0.025, "premium_pct": 0.010},
        {"max_sessions": 4, "tp": 0.35, "sl": 0.30, "gamma_scale": 0.55, "overnight_theta": 0.035, "premium_pct": 0.008},
        {"max_sessions": 3, "tp": 0.30, "sl": 0.20, "gamma_scale": 0.50, "overnight_theta": 0.03, "premium_pct": 0.008},
        {"max_sessions": 2, "tp": 0.35, "sl": 0.25, "gamma_scale": 0.50, "overnight_theta": 0.03, "premium_pct": 0.009},
        {"max_sessions": 3, "tp": 0.45, "sl": 0.28, "gamma_scale": 0.55, "overnight_theta": 0.03, "premium_pct": 0.008},
    ]

    # baseline
    base_sigs = filt(
        cache[(False, True, time(12, 0), 0.15)],
        gap_mode="none",
        min_atr_pct=None,
        skip_friday=False,
    )
    base_ex = exits[1]
    base_pnls = [sim_short_dte(df, s, **base_ex) for s in base_sigs]
    baseline = {"name": "baseline_reclaim_htf_3d", **_stats(base_pnls), **base_ex}
    print(
        f"\nBaseline: n={baseline['n']} WR={baseline['wr']*100:.1f}% "
        f"PF={baseline['pf']:.2f} pnl=${baseline['pnl']:.0f}\n",
        flush=True,
    )

    rows: list[dict] = [baseline]
    total = len(detector_keys) * len(filters) * len(exits)
    print(f"Scoring {total} combos...", flush=True)
    done = 0
    for (confirm, htf, win_end, pierce), (gap_mode, min_atr, skip_fri), ex in itertools.product(
        detector_keys, filters, exits
    ):
        sigs = filt(
            cache[(confirm, htf, win_end, pierce)],
            gap_mode=gap_mode,
            min_atr_pct=min_atr,
            skip_friday=skip_fri,
        )
        pnls = [sim_short_dte(df, s, **ex) for s in sigs]
        st = _stats(pnls)
        rows.append(
            {
                "confirm": confirm,
                "htf": htf,
                "window_end": win_end.isoformat(),
                "pierce": pierce,
                "gap_mode": gap_mode,
                "min_atr_pct": min_atr,
                "skip_friday": skip_fri,
                **ex,
                **st,
                "gate": bool(st["pf"] >= 1.35 and st["n"] >= 30 and st["wr"] >= 0.50),
            }
        )
        done += 1
        if done % 100 == 0:
            print(f"  ... {done}/{total}", flush=True)

    clear_htf_cache()
    configure_amd_params({"require_confirm": False, "apply_htf": False, "window_end": time(12, 0)})

    ranked = sorted(
        [r for r in rows if r["n"] >= 30],
        key=lambda r: (r["pf"] if r["pf"] == r["pf"] else 0, r["exp"], r["n"]),
        reverse=True,
    )
    print("\n=== TOP 15 (n>=30) ===")
    for r in ranked[:15]:
        tag = "GATE" if r.get("gate") else ""
        print(
            f"c={int(r.get('confirm', 0))} end={r.get('window_end','baseline')} "
            f"p={r.get('pierce', 0.15)} gap={r.get('gap_mode','none')} "
            f"atr={r.get('min_atr_pct')} friSkip={r.get('skip_friday')} "
            f"ms={r.get('max_sessions')} tp={r.get('tp')} sl={r.get('sl')} "
            f"| n={r['n']} WR={r['wr']*100:.1f}% PF={r['pf']:.2f} "
            f"exp=${r['exp']:.2f} pnl=${r['pnl']:.0f} {tag}"
        )

    best = ranked[0] if ranked else baseline
    print("\n=== Improvement ===")
    print(f"baseline PF={baseline['pf']:.2f} WR={baseline['wr']*100:.1f}% n={baseline['n']} pnl=${baseline['pnl']:.0f}")
    print(f"best     PF={best['pf']:.2f} WR={best['wr']*100:.1f}% n={best['n']} pnl=${best['pnl']:.0f}")

    out = ROOT / "artifacts" / "amd_short_dte_tune.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    (ROOT / "artifacts" / "amd_short_dte_tune_top.json").write_text(
        json.dumps({"baseline": baseline, "top": ranked[:20]}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
