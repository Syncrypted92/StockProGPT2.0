"""Compare mixed scale-out occupancy: who gets 3 lots vs 2."""

from __future__ import annotations

import pandas as pd

from stockpro.config import load_settings
from stockpro.data import get_spy_5m
from stockpro.spy_day.backtest import ScaleOutConfig, run_spy_day_backtest
from stockpro.spy_day.session import load_spy_day_config


def _so(qbp: dict[str, int]) -> ScaleOutConfig:
    return ScaleOutConfig(qty=3, tp2_pct=0.60, runner_stop_at_entry=True, qty_by_pattern=qbp)


def _row(name, result):
    m = result.metrics
    t = result.trades
    return {
        "variant": name,
        "n": int(m.get("n_trades", 0)),
        "wr": float(m.get("win_rate", 0)),
        "pf": float(m.get("profit_factor", 0)),
        "pnl": float(t["pnl"].sum()) if len(t) else 0.0,
        "orb_n": int((t["pattern"] == "orb").sum()) if len(t) else 0,
        "ph_n": int((t["pattern"] == "power_hour").sum()) if len(t) else 0,
        "amd_n": int((t["pattern"] == "amd").sum()) if len(t) else 0,
        "orb_pnl": float(t.loc[t["pattern"] == "orb", "pnl"].sum()) if len(t) else 0.0,
        "ph_pnl": float(t.loc[t["pattern"] == "power_hour", "pnl"].sum()) if len(t) else 0.0,
        "amd_pnl": float(t.loc[t["pattern"] == "amd", "pnl"].sum()) if len(t) else 0.0,
    }


def main() -> None:
    cfg = load_spy_day_config(load_settings())
    bars = get_spy_5m(load_settings(), refresh=False, rth_only=True)
    variants = {
        "all_3": _so({"orb": 3, "orb_retest": 3, "power_hour": 3, "amd": 3}),
        "live_orb2_ph3": _so({"orb": 2, "orb_retest": 2, "power_hour": 3, "amd": 3}),
        "orb3_ph2": _so({"orb": 3, "orb_retest": 3, "power_hour": 2, "amd": 3}),
        "orb2_ph2": _so({"orb": 2, "orb_retest": 2, "power_hour": 2, "amd": 3}),
    }
    rows = []
    for name, so in variants.items():
        print(name, "...", flush=True)
        r = run_spy_day_backtest(bars, cfg=cfg, scale_out=so)
        rows.append(_row(name, r))
        print(
            f"  n={rows[-1]['n']} WR={rows[-1]['wr']:.1%} PF={rows[-1]['pf']:.2f} "
            f"PnL=${rows[-1]['pnl']:.0f} PH n={rows[-1]['ph_n']} ORB n={rows[-1]['orb_n']}"
        )
    print()
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
