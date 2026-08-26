"""ORB opening-range length research (15 / 30 / 45 / 60 minutes) on SPY 5m."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stockpro.spy_day.backtest import run_spy_day_backtest
from stockpro.spy_day.htf_permission import HtfPermissionConfig
from stockpro.spy_day.session import SpyDayConfig, load_spy_day_config

ROOT = Path(__file__).resolve().parents[1]
OR_MINUTES = [15, 30, 45, 60]


def _cfg(orb_minutes: int) -> SpyDayConfig:
    base = load_spy_day_config()
    htf = HtfPermissionConfig(
        enabled=True,
        skip_4h_counter_trend=True,
        eq_context="none",
        eq_tol_pct=0.0025,
        eq_max_dist_pct=0.004,
        eq_min_touches=2,
    )
    return SpyDayConfig(
        **{
            **{
                k: v
                for k, v in base.__dict__.items()
                if k not in ("htf_permission", "patterns", "orb_minutes", "max_trades_per_day")
            },
            "patterns": ["orb"],
            "orb_minutes": orb_minutes,
            "score_threshold": 0.70,
            "max_trades_per_day": 1,
            "htf_permission": htf,
        }
    )


def main() -> None:
    bars = pd.read_parquet(ROOT / "data" / "bars" / "SPY_5m.parquet")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("America/New_York")
    print(f"ORB range-length research  bars={len(bars)}  {bars.index.min()} -> {bars.index.max()}")
    rows = []
    for m in OR_MINUTES:
        print(f"  OR={m}m ...", flush=True)
        res = run_spy_day_backtest(bars, cfg=_cfg(m))
        met = res.metrics
        pf = float(met.get("profit_factor", 0) or 0)
        if pf != pf:
            pf = 0.0
        n = float(met.get("n_trades", 0))
        row = {
            "orb_minutes": m,
            "n": n,
            "wr": float(met.get("win_rate", 0) or 0),
            "exp": float(met.get("expectancy", 0) or 0),
            "pf": pf,
            "pnl": float(res.trades["pnl"].sum()) if len(res.trades) else 0.0,
            "dd": float(met.get("max_drawdown", 0) or 0),
            "gate": bool(pf >= 1.2 and n >= 40),
        }
        rows.append(row)
        print(
            f"  OR={m:>2}m  n={n:.0f} WR={row['wr']*100:.1f}% PF={pf:.2f} "
            f"exp=${row['exp']:.2f} pnl=${row['pnl']:.0f} gate={row['gate']}"
        )
    out = ROOT / "artifacts" / "orb_range_length_research.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    (ROOT / "artifacts" / "orb_range_length_research.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
