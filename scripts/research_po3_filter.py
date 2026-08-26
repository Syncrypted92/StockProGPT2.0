"""PO3 Judas filter on ORB family — locked rules, full sample + walk-forward.

Compares:
  baseline  = live stack (orb, orb_retest, power_hour, amd) NO po3
  +po3      = same stack, PO3 filter on orb/orb_retest/power_hour only
  orb_only  / orb+po3 for attribution

No parameter search. Promote only if OOS improves without killing n.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from stockpro.config import load_settings
from stockpro.data import get_spy_5m
from stockpro.spy_day.amd import configure_amd_params
from stockpro.spy_day.backtest import run_spy_day_backtest
from stockpro.spy_day.mtf_liquidity import clear_htf_cache
from stockpro.spy_day.po3 import PO3_PARAMS, configure_po3_params
from stockpro.spy_day.session import load_spy_day_config

ROOT = Path(__file__).resolve().parents[1]


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


def _slice_result(result, dates: set):
    tr = result.trades
    if tr is None or len(tr) == 0:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0, "exp": 0.0}
    dcol = pd.to_datetime(tr["date"]).dt.date if "date" in tr.columns else None
    if dcol is None:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0, "exp": 0.0}
    g = tr[dcol.isin(dates)]
    if len(g) == 0:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0, "exp": 0.0}
    pnls = g["pnl"].astype(float)
    wins = float(pnls[pnls > 0].sum())
    losses = float((-pnls[pnls < 0]).sum())
    pf = (wins / losses) if losses > 1e-12 else (float("inf") if wins > 0 else 0.0)
    return {
        "n": int(len(g)),
        "wr": float((pnls > 0).mean()),
        "pf": float(pf) if pf != float("inf") else 99.0,
        "pnl": float(pnls.sum()),
        "exp": float(pnls.mean()),
    }


def _run(label: str, bars, base, *, patterns: list[str], po3: bool):
    cfg = replace(base, patterns=list(patterns))
    configure_amd_params(cfg.amd.detector_dict())
    configure_po3_params(
        {
            "enabled": po3,
            "pierce_atr": 0.15,
            "apply_to": ("orb", "orb_retest", "power_hour"),
            "neutral_policy": "allow",
            "accum_bars": 6,
        }
    )
    r = run_spy_day_backtest(bars, cfg=cfg)
    m = r.metrics
    print(
        f"{label:28s} n={int(m['n_trades']):3d} WR={m['win_rate']*100:5.1f}% "
        f"PF={m['profit_factor']:.2f} E=${m['expectancy']:.2f} pnl=${m['final_equity']-100000:.0f}"
    )
    for p, s in sorted(r.by_pattern.items()):
        print(
            f"  {p:16s} n={int(s['n_trades']):3d} WR={s['win_rate']*100:5.1f}% "
            f"PF={s['profit_factor']:.2f} pnl=${s['total_pnl']:.0f}"
        )
    return r


def main() -> None:
    settings = load_settings()
    bars = get_spy_5m(settings, refresh=False, rth_only=True)
    base = load_spy_day_config(settings)
    live_patterns = list(base.patterns)
    print("=== PO3 filter research (locked Judas rule) ===")
    print(f"bars={len(bars)} {bars.index.min().date()} -> {bars.index.max().date()}")
    print(f"live patterns={live_patterns}")
    print(
        f"PO3 lock: pierce={PO3_PARAMS['pierce_atr']} apply_to={PO3_PARAMS['apply_to']} "
        f"neutral={PO3_PARAMS['neutral_policy']} accum_bars={PO3_PARAMS['accum_bars']}\n"
    )

    print("--- Full sample ---")
    r0 = _run("1) baseline (no PO3)", bars, base, patterns=live_patterns, po3=False)
    r1 = _run("2) +PO3 on ORB family", bars, base, patterns=live_patterns, po3=True)
    r2 = _run("3) ORB family only", bars, base, patterns=["orb", "orb_retest", "power_hour"], po3=False)
    r3 = _run("4) ORB family +PO3", bars, base, patterns=["orb", "orb_retest", "power_hour"], po3=True)

    sess = sorted({ts.date() for ts in bars.index})
    splits = _date_splits(sess, n_folds=3)

    print("\n--- Walk-forward / holdout (live stack ± PO3) ---")
    rows = []
    for name, result, po3 in (
        ("baseline", r0, False),
        ("po3_filter", r1, True),
    ):
        full = {
            "n": int(result.metrics["n_trades"]),
            "wr": float(result.metrics["win_rate"]),
            "pf": float(result.metrics["profit_factor"]),
            "pnl": float(result.metrics["final_equity"] - 100000),
        }
        rows.append({"split": "full", "recipe": name, **full})
        print(f"\n{name} full: n={full['n']} PF={full['pf']:.2f} pnl=${full['pnl']:.0f}")

    # Re-run per fold is expensive; slice trade dates from full runs (same signals chronologically)
    for split_name, train_d, test_d in splits:
        print(f"\n{split_name}: train_days={len(train_d)} test_days={len(test_d)}")
        for name, result in (("baseline", r0), ("po3_filter", r1)):
            tr = _slice_result(result, train_d)
            te = _slice_result(result, test_d)
            print(
                f"  {name:12s} IS n={tr['n']:3d} PF={tr['pf']:.2f} | "
                f"OOS n={te['n']:3d} WR={te['wr']*100:5.1f}% PF={te['pf']:.2f} pnl=${te['pnl']:.0f}"
            )
            rows.append({"split": f"{split_name}_IS", "recipe": name, **tr})
            rows.append({"split": f"{split_name}_OOS", "recipe": name, **te})

    print("\n=== OOS stability (mean fold1-3 OOS PF) ===")
    summary = {}
    for name in ("baseline", "po3_filter"):
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
            "holdout_oos_pnl": hold["pnl"],
        }
        print(
            f"{name:12s} mean_fold_OOS_PF={mpf:.2f}  "
            f"holdout_PF={hold['pf']:.2f} n={hold['n']} pnl=${hold['pnl']:.0f}"
        )

    # Verdict heuristic (not auto-promote)
    b, p = summary["baseline"], summary["po3_filter"]
    better = (
        p["mean_fold_oos_pf"] >= b["mean_fold_oos_pf"] + 0.05
        and p["holdout_oos_pf"] >= b["holdout_oos_pf"]
        and p["holdout_oos_n"] >= max(10, int(0.5 * b["holdout_oos_n"]))
    )
    print("\n=== Verdict ===")
    print(
        "PROMOTE to paper candidate"
        if better
        else "DO NOT PROMOTE — keep PO3 off live (no clear OOS edge / sample risk)"
    )

    configure_po3_params({"enabled": False})
    clear_htf_cache()

    out_csv = ROOT / "artifacts" / "po3_filter_research.csv"
    out_json = ROOT / "artifacts" / "po3_filter_research.json"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_json.write_text(
        json.dumps(
            {
                "summary": summary,
                "promote": better,
                "lock": {
                    "pierce_atr": 0.15,
                    "apply_to": ["orb", "orb_retest", "power_hour"],
                    "neutral_policy": "allow",
                    "accum_bars": 6,
                },
                "full": {
                    "baseline": {
                        "n": int(r0.metrics["n_trades"]),
                        "pf": r0.metrics["profit_factor"],
                        "by_pattern": r0.by_pattern,
                    },
                    "po3": {
                        "n": int(r1.metrics["n_trades"]),
                        "pf": r1.metrics["profit_factor"],
                        "by_pattern": r1.by_pattern,
                    },
                    "orb_only": {
                        "n": int(r2.metrics["n_trades"]),
                        "pf": r2.metrics["profit_factor"],
                    },
                    "orb_po3": {
                        "n": int(r3.metrics["n_trades"]),
                        "pf": r3.metrics["profit_factor"],
                    },
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
