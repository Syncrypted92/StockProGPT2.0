"""Backtest scale-out variants vs hard TP. Live paper is NOT changed.

  2c_hard        — live paper (2 contracts, coded TP 30%/35%)
  2c_hard_tp35   — 2 contracts, hard TP 35% on 0DTE too (size/exit control)
  2c_scale_35_60 — 2 lots: bank 1 at +35%, 2nd to +60% with SL→entry after TP1
  3c_scale       — prior 3-lot idea (coded TP1 / 60% / runner BE)
"""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd

from stockpro.config import ROOT, load_settings
from stockpro.data import get_spy_5m
from stockpro.spy_day.backtest import ScaleOutConfig, run_spy_day_backtest
from stockpro.spy_day.session import load_spy_day_config


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
        "total_return": float(m.get("total_return", 0)),
        "trades_per_day": float(m.get("trades_per_day", 0)),
    }


def _leg_breakdown(trades: pd.DataFrame) -> pd.Series:
    if trades is None or trades.empty or "exit_reason" not in trades.columns:
        return pd.Series(dtype=int)
    return trades["exit_reason"].astype(str).value_counts()


def _print_by_pattern(label: str, result) -> None:
    print(f"{label} by pattern:")
    for pat, stats in sorted(result.by_pattern.items()):
        print(
            f"  {pat}: n={int(stats['n_trades'])} WR={stats['win_rate']:.1%} "
            f"E=${stats['expectancy']:.2f} PF={stats['profit_factor']:.2f} "
            f"PnL=${stats['total_pnl']:.0f}"
        )


def main() -> None:
    settings = load_settings()
    cfg = load_spy_day_config(settings)
    cfg35 = replace(cfg, profit_target_pct=0.35)
    bars = get_spy_5m(settings, refresh=False, rth_only=True)
    if bars.empty:
        raise SystemExit("No 5m bars — run scripts/refresh_spy_bars.py first")

    variants = {
        "2c_hard": run_spy_day_backtest(bars, cfg=cfg, qty=2),
        "2c_hard_tp35": run_spy_day_backtest(bars, cfg=cfg35, qty=2),
        "2c_scale_35_60": run_spy_day_backtest(
            bars,
            cfg=cfg,
            scale_out=ScaleOutConfig(
                qty=2, tp1_pct=0.35, tp2_pct=0.60, runner_stop_at_entry=True
            ),
        ),
        "3c_scale": run_spy_day_backtest(
            bars,
            cfg=cfg,
            scale_out=ScaleOutConfig(qty=3, tp2_pct=0.60, runner_stop_at_entry=True),
        ),
    }

    summary = pd.DataFrame([_row(k, v) for k, v in variants.items()])
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "scale_out_2lot_research.csv"
    trades_path = out_dir / "scale_out_2lot_trades.csv"
    summary.to_csv(summary_path, index=False)
    variants["2c_scale_35_60"].trades.to_csv(trades_path, index=False)

    print("Bars:", len(bars), bars.index.min(), "->", bars.index.max())
    print("Patterns:", cfg.patterns)
    print("Live 0DTE TP/SL:", f"{cfg.profit_target_pct:.0%}/{cfg.stop_loss_pct:.0%}")
    print("Live AMD TP/SL:", f"{cfg.amd.profit_target_pct:.0%}/{cfg.amd.stop_loss_pct:.0%}")
    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("2c_scale_35_60 exit mixes:")
    print(_leg_breakdown(variants["2c_scale_35_60"].trades).to_string())
    print()
    _print_by_pattern("2c_hard", variants["2c_hard"])
    print()
    _print_by_pattern("2c_scale_35_60", variants["2c_scale_35_60"])

    payload = {
        "summary": summary.to_dict(orient="records"),
        "scale_exit_mix": _leg_breakdown(variants["2c_scale_35_60"].trades).to_dict(),
        "by_pattern_2c_hard": variants["2c_hard"].by_pattern,
        "by_pattern_2c_scale": variants["2c_scale_35_60"].by_pattern,
        "notes": {
            "live_unchanged": True,
            "2c_scale": "lot1@35%, lot2@60% with SL moved to entry after TP1",
        },
    }

    def _san(o):
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items()}
        if isinstance(o, float) and o == float("inf"):
            return None
        return o

    json_path = out_dir / "scale_out_2lot_research.json"
    json_path.write_text(json.dumps(_san(payload), indent=2), encoding="utf-8")
    print()
    print("Wrote", summary_path)
    print("Wrote", trades_path)
    print("Wrote", json_path)


if __name__ == "__main__":
    main()
