"""1-year (or max available) ORB + ORB-retest backtest on SPY and QQQ."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stockpro.config import load_settings
from stockpro.data.bars_5m import filter_rth, load_cached_bars_5m, refresh_bars_5m
from stockpro.spy_day.backtest import run_spy_day_backtest
from stockpro.spy_day.htf_permission import HtfPermissionConfig
from stockpro.spy_day.session import SpyDayConfig, load_spy_day_config

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ["SPY", "QQQ"]
HISTORY_DAYS = 365


def _live_cfg() -> SpyDayConfig:
    base = load_spy_day_config()
    return SpyDayConfig(
        **{
            **{k: v for k, v in base.__dict__.items() if k != "htf_permission"},
            "patterns": ["orb", "orb_retest"],
            "score_threshold": 0.70,
            "pattern_min_confidence": {"orb": 0.70, "orb_retest": 0.74},
            "max_trades_per_day": 2,
            "htf_permission": HtfPermissionConfig(
                enabled=True,
                skip_4h_counter_trend=True,
                eq_context="none",
            ),
        }
    )


def _run_symbol(symbol: str, bars: pd.DataFrame) -> dict:
    cfg = _live_cfg()
    res = run_spy_day_backtest(bars, cfg=cfg)
    m = res.metrics
    by = {
        k: {kk: float(vv) for kk, vv in d.items()}
        for k, d in res.by_pattern.items()
    }
    n_days = int(res.trades["date"].nunique()) if len(res.trades) and "date" in res.trades.columns else 0
    return {
        "symbol": symbol,
        "bars": len(bars),
        "start": str(bars.index.min()) if len(bars) else None,
        "end": str(bars.index.max()) if len(bars) else None,
        "n": float(m.get("n_trades", 0)),
        "n_days_traded": n_days,
        "tpd": float(m.get("trades_per_day", 0)),
        "wr": float(m.get("win_rate", 0)),
        "exp": float(m.get("expectancy", 0)),
        "pf": float(m.get("profit_factor", 0) if m.get("profit_factor", 0) == m.get("profit_factor", 0) else 0),
        "pnl": float(res.trades["pnl"].sum()) if len(res.trades) else 0.0,
        "dd": float(m.get("max_drawdown", 0)),
        "gate": bool(m.get("profit_factor", 0) >= 1.2 and m.get("n_trades", 0) >= 40),
        "by_pattern": by,
    }


def main() -> None:
    settings = load_settings()
    settings.require_broker_credentials()
    cfg = _live_cfg()
    print(f"=== 1y SPY+QQQ backtest | patterns={cfg.patterns} | days={HISTORY_DAYS} ===\n")

    rows = []
    for sym in SYMBOLS:
        print(f"--- refresh {sym} ---")
        refresh_bars_5m(settings, symbol=sym, history_days=HISTORY_DAYS, force_full=True)
        bars = filter_rth(load_cached_bars_5m(settings, sym))
        print(f"{sym} RTH bars={len(bars)}  {bars.index.min()} -> {bars.index.max()}")
        r = _run_symbol(sym, bars)
        rows.append(r)
        print(
            f"{sym}: n={r['n']:.0f} tpd={r['tpd']:.2f} WR={r['wr']:.1%} "
            f"E=${r['exp']:.2f} PF={r['pf']:.2f} PnL=${r['pnl']:.0f} gate={r['gate']}"
        )
        for pat, stats in r["by_pattern"].items():
            print(
                f"    {pat}: n={stats.get('n_trades', 0):.0f} "
                f"WR={stats.get('win_rate', 0):.1%} "
                f"E=${stats.get('expectancy', 0):.2f} "
                f"PF={stats.get('profit_factor', 0):.2f}"
            )
        print()

    # Combined view: independent books (same rules, no shared position limit)
    combined = {
        "n": sum(r["n"] for r in rows),
        "pnl": sum(r["pnl"] for r in rows),
        "wr_avg": sum(r["wr"] * r["n"] for r in rows) / max(sum(r["n"] for r in rows), 1),
        "tpd_sum": sum(r["tpd"] for r in rows),
        "symbols": [r["symbol"] for r in rows],
        "note": "Independent per-symbol books; tpd_sum ≈ expected trades/day if both run",
    }
    print("=== Combined (independent SPY + QQQ books) ===")
    print(
        f"n={combined['n']:.0f}  weighted WR={combined['wr_avg']:.1%}  "
        f"PnL=${combined['pnl']:.0f}  tpd_sum~={combined['tpd_sum']:.2f}"
    )

    out = {
        "history_days_requested": HISTORY_DAYS,
        "patterns": cfg.patterns,
        "per_symbol": rows,
        "combined": combined,
    }
    path = ROOT / "artifacts" / "spy_qqq_1y_backtest.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
