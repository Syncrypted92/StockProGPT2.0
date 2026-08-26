"""Backtest OTE variants alone and combined with ORB+HTF."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import time
from pathlib import Path

import pandas as pd

from stockpro.spy_day.backtest import run_spy_day_backtest
from stockpro.spy_day.htf_permission import HtfPermissionConfig
from stockpro.spy_day.patterns import OTE_PARAMS
from stockpro.spy_day.session import SpyDayConfig, load_spy_day_config

ROOT = Path(__file__).resolve().parents[1]


VARIANTS: list[tuple[str, dict]] = [
    (
        "ote_good_confirm",
        {
            "window_start": time(10, 0),
            "window_end": time(14, 0),
            "require_bos": False,
            "require_ob": False,
            "min_grade": "good",
            "one_per_day": True,
            "require_confirmation": True,
            "killzone_only": True,
        },
    ),
    (
        "ote_better_ob",
        {
            "window_start": time(10, 0),
            "window_end": time(14, 0),
            "require_bos": False,
            "require_ob": True,
            "min_grade": "better",
            "one_per_day": True,
            "require_confirmation": True,
            "killzone_only": True,
        },
    ),
    (
        "ote_best_ob_fvg",
        {
            "window_start": time(10, 0),
            "window_end": time(14, 0),
            "require_bos": True,
            "require_ob": True,
            "min_grade": "best",
            "one_per_day": True,
            "require_confirmation": True,
            "killzone_only": True,
        },
    ),
    (
        "ote_good_wide",
        {
            "window_start": time(10, 0),
            "window_end": time(14, 30),
            "require_bos": False,
            "require_ob": False,
            "min_grade": "good",
            "one_per_day": True,
            "require_confirmation": True,
            "killzone_only": False,
        },
    ),
]


def _apply(params: dict) -> None:
    OTE_PARAMS.clear()
    OTE_PARAMS.update(params)


def _run(bars: pd.DataFrame, patterns: list[str], name: str) -> dict:
    base = load_spy_day_config()
    cfg = SpyDayConfig(
        **{
            **{k: v for k, v in base.__dict__.items() if k != "htf_permission"},
            "patterns": patterns,
            "score_threshold": 0.70,
            "pattern_min_confidence": {"orb": 0.70, "ote": 0.72},
            "max_trades_per_day": 2,
            "htf_permission": base.htf_permission
            if "orb" in patterns
            else HtfPermissionConfig(enabled=False),
        }
    )
    # For OTE-only, disable HTF on ORB path (unused)
    if patterns == ["ote"]:
        cfg.htf_permission = HtfPermissionConfig(enabled=False)

    res = run_spy_day_backtest(bars, cfg=cfg)
    m = res.metrics
    by = res.by_pattern
    return {
        "name": name,
        "patterns": patterns,
        "n": m.get("n_trades", 0),
        "wr": m.get("win_rate", 0),
        "exp": m.get("expectancy", 0),
        "pf": m.get("profit_factor", 0),
        "pnl": float(res.trades["pnl"].sum()) if len(res.trades) else 0.0,
        "dd": m.get("max_drawdown", 0),
        "by_pattern": by,
        "gate": bool(
            m.get("profit_factor", 0) >= base.backtest_gate_pf
            and m.get("n_trades", 0) >= base.backtest_gate_min_trades
        ),
    }


def main() -> None:
    path = ROOT / "data" / "bars" / "SPY_5m.parquet"
    bars = pd.read_parquet(path)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("America/New_York")

    rows: list[dict] = []
    defaults = deepcopy(OTE_PARAMS)

    # Live baseline: ORB + HTF
    _apply(VARIANTS[0][1])
    rows.append(_run(bars, ["orb"], "baseline_orb_htf"))

    for vname, params in VARIANTS:
        _apply(params)
        rows.append(_run(bars, ["ote"], f"ote_only:{vname}"))
        rows.append(_run(bars, ["orb", "ote"], f"orb+ote:{vname}"))

    _apply(defaults)

    def pf_rank(x: float) -> float:
        if x != x or x == float("inf"):
            return 0.0
        return float(x)

    print("=== OTE research (SPY 5m ~6mo) ===")
    for r in rows:
        ote = r["by_pattern"].get("ote", {})
        orb = r["by_pattern"].get("orb", {})
        print(
            f"{r['name'][:42]:42s} n={r['n']:5.0f} WR={r['wr']:5.1%} "
            f"E=${r['exp']:7.2f} PF={r['pf']:5.2f} PnL=${r['pnl']:7.0f} gate={r['gate']}"
            f"  | ote_n={ote.get('n_trades', 0):.0f} ote_WR={ote.get('win_rate', 0):.0%}"
            f" orb_n={orb.get('n_trades', 0):.0f}"
        )

    out = ROOT / "artifacts" / "ote_research.json"
    # Strip non-serializable
    serial = []
    for r in rows:
        serial.append(
            {
                **{k: v for k, v in r.items() if k != "by_pattern"},
                "by_pattern": {
                    k: {kk: float(vv) if isinstance(vv, (int, float)) else vv for kk, vv in d.items()}
                    for k, d in r["by_pattern"].items()
                },
            }
        )
    out.write_text(json.dumps({"variants": serial}, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
