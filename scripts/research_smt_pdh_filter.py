"""SMT + PDH/PDL locked filters — separate then combined, + walk-forward.

No parameter search. All filters default off in live config.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from stockpro.config import load_settings
from stockpro.data import get_spy_5m, refresh_bars_5m
from stockpro.spy_day.amd import configure_amd_params
from stockpro.spy_day.backtest import run_spy_day_backtest
from stockpro.spy_day.mtf_liquidity import clear_htf_cache
from stockpro.spy_day.pdh_filter import configure_pdh_params
from stockpro.spy_day.po3 import configure_po3_params
from stockpro.spy_day.session import load_spy_day_config
from stockpro.spy_day.smt import clear_smt_cache, configure_smt_params

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


def _slice_result(result, dates: set) -> dict:
    tr = result.trades
    if tr is None or len(tr) == 0 or "date" not in tr.columns:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0, "exp": 0.0}
    dcol = pd.to_datetime(tr["date"]).dt.date
    g = tr[dcol.isin(dates)]
    if len(g) == 0:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0, "exp": 0.0}
    pnls = g["pnl"].astype(float)
    wins = float(pnls[pnls > 0].sum())
    losses = float((-pnls[pnls < 0]).sum())
    pf = (wins / losses) if losses > 1e-12 else (99.0 if wins > 0 else 0.0)
    return {
        "n": int(len(g)),
        "wr": float((pnls > 0).mean()),
        "pf": float(pf),
        "pnl": float(pnls.sum()),
        "exp": float(pnls.mean()),
    }


def _set_filters(*, smt: bool, pdh: bool) -> None:
    configure_po3_params({"enabled": False})
    configure_smt_params(
        {
            "enabled": smt,
            "min_div": 0.0008,
            "apply_to": ("orb", "orb_retest", "power_hour", "amd"),
        }
    )
    configure_pdh_params(
        {
            "enabled": pdh,
            "pierce_atr": 0.15,
            "apply_to": ("orb", "orb_retest", "power_hour", "amd"),
        }
    )
    clear_smt_cache()


def _run(label: str, bars, base, *, smt: bool, pdh: bool):
    cfg = replace(base, patterns=list(base.patterns))
    configure_amd_params(cfg.amd.detector_dict())
    _set_filters(smt=smt, pdh=pdh)
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


def _ensure_qqq(settings) -> None:
    path = ROOT / "data" / "bars" / "QQQ_5m.parquet"
    if path.exists() and path.stat().st_size > 1000:
        print(f"QQQ bars OK: {path}")
        return
    print("Downloading QQQ 5m bars for SMT...")
    refresh_bars_5m(settings, symbol="QQQ", force_full=True)


def main() -> None:
    settings = load_settings()
    _ensure_qqq(settings)
    bars = get_spy_5m(settings, refresh=False, rth_only=True)
    base = load_spy_day_config(settings)

    print("=== SMT + PDH/PDL filter research (locked) ===")
    print(f"bars={len(bars)} {bars.index.min().date()} -> {bars.index.max().date()}")
    print(f"patterns={base.patterns}\n")

    print("--- Full sample ---")
    recipes = {
        "baseline": _run("1) baseline", bars, base, smt=False, pdh=False),
        "smt": _run("2) SMT only", bars, base, smt=True, pdh=False),
        "pdh": _run("3) PDH/PDL only", bars, base, smt=False, pdh=True),
        "smt_pdh": _run("4) SMT + PDH/PDL", bars, base, smt=True, pdh=True),
    }

    sess = sorted({ts.date() for ts in bars.index})
    splits = _date_splits(sess, n_folds=3)
    rows: list[dict] = []

    print("\n--- Walk-forward / holdout ---")
    for name, result in recipes.items():
        full = {
            "n": int(result.metrics["n_trades"]),
            "wr": float(result.metrics["win_rate"]),
            "pf": float(result.metrics["profit_factor"]),
            "pnl": float(result.metrics["final_equity"] - 100000),
        }
        rows.append({"split": "full", "recipe": name, **full})

    for split_name, train_d, test_d in splits:
        print(f"\n{split_name}: train_days={len(train_d)} test_days={len(test_d)}")
        for name, result in recipes.items():
            tr = _slice_result(result, train_d)
            te = _slice_result(result, test_d)
            print(
                f"  {name:10s} IS n={tr['n']:3d} PF={tr['pf']:.2f} | "
                f"OOS n={te['n']:3d} WR={te['wr']*100:5.1f}% PF={te['pf']:.2f} pnl=${te['pnl']:.0f}"
            )
            rows.append({"split": f"{split_name}_IS", "recipe": name, **tr})
            rows.append({"split": f"{split_name}_OOS", "recipe": name, **te})

    print("\n=== OOS stability (mean fold1-3 OOS PF; holdout) ===")
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
            "holdout_oos_pnl": hold["pnl"],
            "full_pf": next(r["pf"] for r in rows if r["recipe"] == name and r["split"] == "full"),
            "full_n": next(r["n"] for r in rows if r["recipe"] == name and r["split"] == "full"),
        }
        print(
            f"{name:10s} full_PF={summary[name]['full_pf']:.2f} n={summary[name]['full_n']} | "
            f"mean_fold_OOS_PF={mpf:.2f} holdout_PF={hold['pf']:.2f} n={hold['n']} pnl=${hold['pnl']:.0f}"
        )

    base_s = summary["baseline"]

    def _better(name: str) -> bool:
        s = summary[name]
        return (
            s["mean_fold_oos_pf"] >= base_s["mean_fold_oos_pf"] + 0.05
            and s["holdout_oos_pf"] >= base_s["holdout_oos_pf"]
            and s["holdout_oos_n"] >= max(8, int(0.4 * base_s["holdout_oos_n"]))
        )

    print("\n=== Verdict (vs baseline; rule-of-thumb only) ===")
    for name in ("smt", "pdh", "smt_pdh"):
        flag = "CANDIDATE" if _better(name) else "do not promote"
        print(f"  {name:10s} -> {flag}")

    _set_filters(smt=False, pdh=False)
    clear_htf_cache()
    clear_smt_cache()

    out_csv = ROOT / "artifacts" / "smt_pdh_filter_research.csv"
    out_json = ROOT / "artifacts" / "smt_pdh_filter_research.json"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_json.write_text(
        json.dumps({"summary": summary, "promote": {n: _better(n) for n in ("smt", "pdh", "smt_pdh")}}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
