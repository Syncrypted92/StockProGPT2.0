"""Backtest ICT NY macros as an entry filter. Live paper is NOT changed.

Windows (ET):
  09:50–10:10  AM macro (ORB)
  10:50–11:10  late-morning (AMD)
  13:50–14:10  afternoon (touches power hour)

All runs use 3-lot scale-out + live patterns.
"""

from __future__ import annotations

import json
from datetime import time

import pandas as pd

from stockpro.config import ROOT, load_settings
from stockpro.data import get_spy_5m
from stockpro.spy_day.backtest import ScaleOutConfig, run_spy_day_backtest
from stockpro.spy_day.session import load_spy_day_config

MACROS = [
    (time(9, 50), time(10, 10), "am"),
    (time(10, 50), time(11, 10), "late_am"),
    (time(13, 50), time(14, 10), "pm"),
]


def _in_windows(t: time, windows: list[tuple[time, time, str]]) -> bool:
    for a, b, _ in windows:
        if a <= t <= b:
            return True
    return False


def _bar_time(ts) -> time:
    t = ts.timetz().replace(tzinfo=None) if hasattr(ts, "timetz") else ts.time()
    return t


def make_skip(windows: list[tuple[time, time, str]], patterns: set[str] | None = None):
    def _fn(sig, i, df) -> bool:
        if patterns is not None and sig.pattern not in patterns:
            return False
        return not _in_windows(_bar_time(df.index[i]), windows)

    return _fn


def _row(name: str, result) -> dict:
    m = result.metrics
    trades = result.trades
    return {
        "variant": name,
        "n_trades": int(m.get("n_trades", 0)),
        "win_rate": float(m.get("win_rate", 0)),
        "expectancy": float(m.get("expectancy", 0)),
        "profit_factor": float(m.get("profit_factor", 0)),
        "total_pnl": float(trades["pnl"].sum()) if len(trades) else 0.0,
        "max_drawdown": float(m.get("max_drawdown", 0)),
    }


def _print_by_pattern(label: str, result) -> None:
    print(f"{label} by pattern:")
    for pat, stats in sorted(result.by_pattern.items()):
        print(
            f"  {pat}: n={int(stats['n_trades'])} WR={stats['win_rate']:.1%} "
            f"E=${stats['expectancy']:.2f} PF={stats['profit_factor']:.2f} "
            f"PnL=${stats['total_pnl']:.0f}"
        )


def _macro_hit_rate(trades: pd.DataFrame) -> None:
    if trades is None or trades.empty:
        return
    ts = pd.to_datetime(trades["datetime"], utc=True)
    t = ts.dt.tz_convert("America/New_York").dt.time if ts.dt.tz is not None else ts.dt.time
    hits = t.map(lambda x: _in_windows(x, MACROS))
    print(
        f"Baseline entries already in a macro: {int(hits.sum())}/{len(trades)} "
        f"({hits.mean():.1%})"
    )
    for a, b, name in MACROS:
        n = int(t.map(lambda x, a=a, b=b: a <= x <= b).sum())
        print(f"  {name} {a.strftime('%H:%M')}–{b.strftime('%H:%M')}: {n}")


def main() -> None:
    settings = load_settings()
    cfg = load_spy_day_config(settings)
    bars = get_spy_5m(settings, refresh=False, rth_only=True)
    if bars.empty:
        raise SystemExit("No 5m bars")
    scale = ScaleOutConfig(qty=3, tp2_pct=0.60, runner_stop_at_entry=True)
    am = [MACROS[0]]
    am_late = MACROS[:2]
    all_m = MACROS

    print("Bars:", len(bars), bars.index.min(), "->", bars.index.max())
    variants = {}
    print("baseline...")
    variants["3c_scale"] = run_spy_day_backtest(bars, cfg=cfg, scale_out=scale)
    print("ORB/retest only in 09:50–10:10...")
    variants["orb_am_macro"] = run_spy_day_backtest(
        bars,
        cfg=cfg,
        scale_out=scale,
        skip_fn=make_skip(am, {"orb", "orb_retest"}),
    )
    print("ORB/retest in 09:50–10:10 or 10:50–11:10...")
    variants["orb_am_late"] = run_spy_day_backtest(
        bars,
        cfg=cfg,
        scale_out=scale,
        skip_fn=make_skip(am_late, {"orb", "orb_retest"}),
    )
    print("All 0DTE (orb/retest/PH) in any of 3 macros...")
    variants["0dte_all_macros"] = run_spy_day_backtest(
        bars,
        cfg=cfg,
        scale_out=scale,
        skip_fn=make_skip(all_m, {"orb", "orb_retest", "power_hour"}),
    )
    print("Every pattern including AMD in any of 3 macros...")
    variants["all_in_macros"] = run_spy_day_backtest(
        bars,
        cfg=cfg,
        scale_out=scale,
        skip_fn=make_skip(all_m, None),
    )

    summary = pd.DataFrame([_row(k, v) for k, v in variants.items()])
    out = ROOT / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "ict_macro_research.csv", index=False)

    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    _macro_hit_rate(variants["3c_scale"].trades)
    print()
    for name, res in variants.items():
        _print_by_pattern(name, res)
        print()

    payload = {
        "summary": summary.to_dict(orient="records"),
        "macros": [f"{a.strftime('%H:%M')}-{b.strftime('%H:%M')} {n}" for a, b, n in MACROS],
        "live_unchanged": True,
    }

    def _san(o):
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items()}
        if isinstance(o, float) and o == float("inf"):
            return None
        return o

    (out / "ict_macro_research.json").write_text(json.dumps(_san(payload), indent=2), encoding="utf-8")
    print("Wrote", out / "ict_macro_research.csv")


if __name__ == "__main__":
    main()
