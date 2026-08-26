"""Walk-forward / OOS check for AMD short-DTE paper recipe (anti-overfit).

Locks the live-ish recipes BEFORE seeing test folds:
  A) simple: reclaim + HTF + 10-12 + pierce 0.15 + TP35/SL30  (incl Friday)
  B) live:   A + skip Friday
  C) lean:   reclaim + HTF + 10-12 + pierce 0.15 + TP30/SL25 (incl Friday)

No new knobs are searched — only chronological stability.
"""

from __future__ import annotations

import json
from datetime import time
from pathlib import Path

import pandas as pd

from research_amd_short_dte_tune import _stats, filt, precompute, sim_short_dte
from stockpro.spy_day.mtf_liquidity import clear_htf_cache, set_htf_cache
from stockpro.spy_day.patterns import enrich_bars

ROOT = Path(__file__).resolve().parents[1]

EXIT = {
    "max_sessions": 3,
    "tp": 0.35,
    "sl": 0.30,
    "gamma_scale": 0.55,
    "overnight_theta": 0.03,
    "premium_pct": 0.008,
}
EXIT_LEAN = {**EXIT, "tp": 0.30, "sl": 0.25}


def _date_splits(dates: list, n_folds: int = 3) -> list[tuple[str, set, set]]:
    """Expanding train → next fold test."""
    dates = sorted(dates)
    n = len(dates)
    fold_size = max(n // (n_folds + 1), 1)
    out = []
    for k in range(1, n_folds + 1):
        train_end = fold_size * k
        test_end = min(fold_size * (k + 1), n) if k < n_folds else n
        if test_end <= train_end:
            continue
        train = set(dates[:train_end])
        test = set(dates[train_end:test_end])
        out.append((f"fold{k}", train, test))
    cut = int(n * 0.60)
    out.append(("holdout_60_40", set(dates[:cut]), set(dates[cut:])))
    return out


def _eval(df: pd.DataFrame, sigs: list[dict], dates: set, exit_kw: dict) -> dict:
    use = [s for s in sigs if s["ts"].date() in dates]
    pnls = [sim_short_dte(df, s, **exit_kw) for s in use]
    return _stats(pnls)


def main() -> None:
    bars = pd.read_parquet(ROOT / "data" / "bars" / "SPY_5m.parquet")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("America/New_York")
    df = enrich_bars(bars, orb_minutes=30)
    set_htf_cache(df)

    print("=== AMD walk-forward (short-DTE proxy) ===")
    print(f"bars={len(df)}  {df.index.min().date()} -> {df.index.max().date()}")

    raw = precompute(df, confirm=False, htf=True, win_end=time(12, 0), pierce=0.15)
    recipes = {
        "A_simple_incl_fri": {
            "sigs": filt(raw, gap_mode="none", min_atr_pct=None, skip_friday=False),
            "exit": EXIT,
        },
        "B_live_skip_fri": {
            "sigs": filt(raw, gap_mode="none", min_atr_pct=None, skip_friday=True),
            "exit": EXIT,
        },
        "C_lean_tp30_sl25": {
            "sigs": filt(raw, gap_mode="none", min_atr_pct=None, skip_friday=False),
            "exit": EXIT_LEAN,
        },
    }

    sess = sorted({ts.date() for ts in df.index})
    splits = _date_splits(sess, n_folds=3)

    rows: list[dict] = []
    print("\n--- Full sample ---")
    for name, r in recipes.items():
        st = _stats([sim_short_dte(df, s, **r["exit"]) for s in r["sigs"]])
        print(f"{name:22s} n={st['n']:3d} WR={st['wr']*100:5.1f}% PF={st['pf']:.2f} pnl=${st['pnl']:.0f}")
        rows.append({"split": "full", "recipe": name, **st})

    print("\n--- Walk-forward / holdout ---")
    for split_name, train_d, test_d in splits:
        print(f"\n{split_name}: train_days={len(train_d)} test_days={len(test_d)}")
        for name, r in recipes.items():
            tr = _eval(df, r["sigs"], train_d, r["exit"])
            te = _eval(df, r["sigs"], test_d, r["exit"])
            print(
                f"  {name:22s} "
                f"IS n={tr['n']:3d} PF={tr['pf']:.2f} | "
                f"OOS n={te['n']:3d} WR={te['wr']*100:5.1f}% PF={te['pf']:.2f} pnl=${te['pnl']:.0f}"
            )
            rows.append({"split": f"{split_name}_IS", "recipe": name, **tr})
            rows.append({"split": f"{split_name}_OOS", "recipe": name, **te})

    clear_htf_cache()

    print("\n=== OOS stability (mean of fold1-3 OOS PF; holdout separate) ===")
    summary = {}
    for name in recipes:
        fold_oos = [
            r
            for r in rows
            if r["recipe"] == name and r["split"].endswith("_OOS") and r["split"].startswith("fold")
        ]
        hold = next(r for r in rows if r["recipe"] == name and r["split"] == "holdout_60_40_OOS")
        mpf = float(pd.Series([x["pf"] for x in fold_oos if x["n"] > 0]).mean()) if fold_oos else 0.0
        summary[name] = {
            "mean_fold_oos_pf": mpf,
            "holdout_oos_pf": hold["pf"],
            "holdout_oos_n": hold["n"],
            "holdout_oos_wr": hold["wr"],
            "holdout_oos_pnl": hold["pnl"],
        }
        print(
            f"{name:22s} mean_fold_OOS_PF={mpf:.2f}  "
            f"holdout_PF={hold['pf']:.2f} n={hold['n']} WR={hold['wr']*100:.1f}%"
        )

    out_csv = ROOT / "artifacts" / "amd_walk_forward.csv"
    out_json = ROOT / "artifacts" / "amd_walk_forward.json"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
