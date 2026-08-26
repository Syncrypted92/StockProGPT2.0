"""Compare lot-3 runner exits vs live BE (research only — does not change paper).

Live is 5m manage. We do not have historical 1m option mids; this uses the same
5m premium proxy as the scale-out sample. 1m live would mainly cut gap-through
on SL/BE; trail/TP3 is what banks the runner run.
"""

from __future__ import annotations

from stockpro.config import ROOT, load_settings
from stockpro.data import get_spy_5m
from stockpro.spy_day.backtest import ScaleOutConfig, run_spy_day_backtest
from stockpro.spy_day.session import load_spy_day_config


def _row(name: str, result) -> dict:
    m = result.metrics
    trades = result.trades
    reasons = (
        trades["exit_reason"].astype(str).value_counts().to_dict() if len(trades) else {}
    )
    return {
        "variant": name,
        "n": int(m.get("n_trades", 0)),
        "wr": float(m.get("win_rate", 0)),
        "pf": float(m.get("profit_factor", 0)),
        "exp": float(m.get("expectancy", 0)),
        "pnl": float(trades["pnl"].sum()) if len(trades) else 0.0,
        "dd": float(m.get("max_drawdown", 0)),
        "reasons": reasons,
    }


def main() -> None:
    settings = load_settings()
    cfg = load_spy_day_config(settings)
    bars = get_spy_5m(settings, refresh=False, rth_only=True)
    if bars.empty:
        raise SystemExit("No 5m bars")

    live = ScaleOutConfig(qty=3, tp2_pct=0.60, runner_stop_at_entry=True)
    variants = {
        "live_be": live,
        "trail_12": ScaleOutConfig(qty=3, tp2_pct=0.60, runner_stop_at_entry=True, runner_trail_pct=0.12),
        "trail_15": ScaleOutConfig(qty=3, tp2_pct=0.60, runner_stop_at_entry=True, runner_trail_pct=0.15),
        "trail_20": ScaleOutConfig(qty=3, tp2_pct=0.60, runner_stop_at_entry=True, runner_trail_pct=0.20),
        "trail_25": ScaleOutConfig(qty=3, tp2_pct=0.60, runner_stop_at_entry=True, runner_trail_pct=0.25),
        "tp3_80": ScaleOutConfig(qty=3, tp2_pct=0.60, tp3_pct=0.80, runner_stop_at_entry=True),
        "tp3_100": ScaleOutConfig(qty=3, tp2_pct=0.60, tp3_pct=1.00, runner_stop_at_entry=True),
        "tp3_100_trail_20": ScaleOutConfig(
            qty=3, tp2_pct=0.60, tp3_pct=1.00, runner_trail_pct=0.20, runner_stop_at_entry=True
        ),
    }

    rows = []
    print(f"{'variant':22} {'n':>4} {'WR':>7} {'PF':>6} {'exp':>8} {'pnl':>9} {'dd':>8}")
    for name, so in variants.items():
        res = run_spy_day_backtest(bars, cfg=cfg, scale_out=so)
        r = _row(name, res)
        rows.append(r)
        print(
            f"{name:22} {r['n']:4d} {r['wr']:6.1%} {r['pf']:6.2f} "
            f"{r['exp']:8.2f} {r['pnl']:9.0f} {r['dd']:8.2%}"
        )
        extra = {k: v for k, v in r["reasons"].items() if "trail" in k or "target_3" in k or "breakeven" in k}
        if extra:
            print(f"  {extra}")

    import pandas as pd

    out = ROOT / "artifacts" / "runner_exit_research.csv"
    pd.DataFrame([{k: v for k, v in r.items() if k != "reasons"} for r in rows]).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
