"""Backtest locked 4H sweep+engulf (London/NY) — 0DTE + short-DTE + walk-forward.

Live stack unchanged. Research only.
"""

from __future__ import annotations

import json
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from stockpro.config import load_settings
from stockpro.data import get_spy_5m
from stockpro.spy_day.backtest import _simulate_0dte_path, _simulate_short_dte_path
from stockpro.spy_day.h4_sweep_engulf import collect_h4_sweep_engulf
from stockpro.spy_day.patterns import enrich_bars

ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")


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


def _date_splits(dates: list, n_folds: int = 3) -> list[tuple[str, set, set]]:
    dates = sorted(dates)
    n = len(dates)
    fold_size = max(n // (n_folds + 1), 1)
    out = []
    for k in range(1, n_folds + 1):
        train_end = fold_size * k
        test_end = min(fold_size * (k + 1), n) if k < n_folds else n
        if test_end <= train_end:
            continue
        out.append((f"fold{k}", set(dates[:train_end]), set(dates[train_end:test_end])))
    cut = int(n * 0.60)
    out.append(("holdout_60_40", set(dates[:cut]), set(dates[cut:])))
    return out


def _sim_vehicle(
    df: pd.DataFrame,
    sigs: list,
    *,
    vehicle: str,
) -> list[dict]:
    closes = df["Close"]
    trades = []
    force_flat = time(15, 45)
    for sig in sigs:
        ts = sig.ts
        if ts not in closes.index:
            # snap to nearest bar
            pos = closes.index.searchsorted(ts)
            if pos >= len(closes.index):
                continue
            ts = closes.index[pos]
        spot = float(sig.spot)
        side = 1 if sig.side == "call" else -1
        t = ts.timetz().replace(tzinfo=None) if hasattr(ts, "timetz") else ts.time()

        if vehicle == "0dte":
            # SPY 0DTE needs RTH
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
        trades.append(
            {
                "ts": ts,
                "date": ts.date(),
                "side": sig.side,
                "session": sig.session,
                "reason": sig.reason,
                "entry": entry,
                "exit": exit_prem,
                "pnl": pnl,
                "exit_reason": reason,
                "vehicle": vehicle,
            }
        )
    return trades


def _slice(trades: list[dict], dates: set) -> dict:
    pnls = [t["pnl"] for t in trades if t["date"] in dates]
    return _stats(pnls)


def main() -> None:
    settings = load_settings()
    # Extended hours so London / overnight 4H exist
    bars = get_spy_5m(settings, refresh=False, rth_only=False)
    if bars is None or bars.empty:
        raise SystemExit("No extended SPY bars — run scripts/refresh_spy_bars.py --full")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize(ET)
    else:
        bars.index = bars.index.tz_convert(ET)

    enriched = enrich_bars(bars, orb_minutes=30)
    sigs = collect_h4_sweep_engulf(enriched)
    print("=== 4H sweep + engulf (locked) ===")
    print(f"bars={len(enriched)} {enriched.index.min()} -> {enriched.index.max()}")
    print(f"signals={len(sigs)}  london={sum(1 for s in sigs if s.session=='london')} ny={sum(1 for s in sigs if s.session=='ny')}")

    rows = []
    recipes = {}
    for vehicle in ("0dte", "short_dte"):
        trades = _sim_vehicle(enriched, sigs, vehicle=vehicle)
        recipes[vehicle] = trades
        st = _stats([t["pnl"] for t in trades])
        by_sess = {}
        for sess in ("london", "ny"):
            by_sess[sess] = _stats([t["pnl"] for t in trades if t["session"] == sess])
        print(
            f"\n{vehicle:10s} n={st['n']:3d} WR={st['wr']*100:5.1f}% "
            f"PF={st['pf']:.2f} E=${st['exp']:.2f} pnl=${st['pnl']:.0f}"
        )
        for sess, ss in by_sess.items():
            print(
                f"  {sess:7s} n={ss['n']:3d} WR={ss['wr']*100:5.1f}% "
                f"PF={ss['pf']:.2f} pnl=${ss['pnl']:.0f}"
            )
        rows.append({"split": "full", "recipe": vehicle, **st})
        for sess, ss in by_sess.items():
            rows.append({"split": "full", "recipe": f"{vehicle}_{sess}", **ss})

    # Walk-forward on signal calendar days
    sig_days = sorted({s.ts.date() for s in sigs})
    # Prefer full bar calendar for chronology
    sess_days = sorted({ts.date() for ts in enriched.index})
    splits = _date_splits(sess_days, n_folds=3)

    print("\n--- Walk-forward / holdout ---")
    for split_name, train_d, test_d in splits:
        print(f"\n{split_name}: train_days={len(train_d)} test_days={len(test_d)}")
        for vehicle, trades in recipes.items():
            tr = _slice(trades, train_d)
            te = _slice(trades, test_d)
            print(
                f"  {vehicle:10s} IS n={tr['n']:3d} PF={tr['pf']:.2f} | "
                f"OOS n={te['n']:3d} WR={te['wr']*100:5.1f}% PF={te['pf']:.2f} pnl=${te['pnl']:.0f}"
            )
            rows.append({"split": f"{split_name}_IS", "recipe": vehicle, **tr})
            rows.append({"split": f"{split_name}_OOS", "recipe": vehicle, **te})

    print("\n=== OOS stability ===")
    summary = {}
    for vehicle in ("0dte", "short_dte"):
        fold_oos = [
            r
            for r in rows
            if r["recipe"] == vehicle and r["split"].endswith("_OOS") and r["split"].startswith("fold")
        ]
        hold = next(r for r in rows if r["recipe"] == vehicle and r["split"] == "holdout_60_40_OOS")
        full = next(r for r in rows if r["recipe"] == vehicle and r["split"] == "full")
        mpf = float(pd.Series([x["pf"] for x in fold_oos if x["n"] > 0]).mean()) if fold_oos else 0.0
        summary[vehicle] = {
            "full": full,
            "mean_fold_oos_pf": mpf,
            "holdout": hold,
            "n_signals_total": len(sigs),
        }
        print(
            f"{vehicle:10s} full_PF={full['pf']:.2f} n={full['n']} | "
            f"mean_fold_OOS_PF={mpf:.2f} holdout_PF={hold['pf']:.2f} n={hold['n']} pnl=${hold['pnl']:.0f}"
        )

    # Soft verdict vs random / dead
    print("\n=== Verdict ===")
    for vehicle, s in summary.items():
        ok = (
            s["full"]["n"] >= 20
            and s["full"]["pf"] >= 1.2
            and s["mean_fold_oos_pf"] >= 1.1
            and s["holdout"]["pf"] >= 1.0
            and s["holdout"]["n"] >= 8
        )
        print(
            f"  {vehicle:10s} -> {'interesting research candidate' if ok else 'weak / do not promote (live stays off)'}"
        )

    out_csv = ROOT / "artifacts" / "h4_sweep_engulf_research.csv"
    out_json = ROOT / "artifacts" / "h4_sweep_engulf_research.json"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    # sample signals
    sample = [
        {
            "ts": str(s.ts),
            "side": s.side,
            "session": s.session,
            "reason": s.reason,
            "spot": s.spot,
        }
        for s in sigs[:30]
    ]
    out_json.write_text(
        json.dumps({"summary": summary, "sample_signals": sample, "n_signals": len(sigs)}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
