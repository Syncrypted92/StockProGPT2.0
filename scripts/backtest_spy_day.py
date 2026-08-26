"""Backtest SPY 0DTE day patterns on 5m bars; write report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from stockpro.config import ROOT, load_settings
from stockpro.data import get_spy_5m
from stockpro.spy_day.backtest import run_spy_day_backtest
from stockpro.spy_day.session import load_spy_day_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest SPY day 0DTE patterns")
    parser.add_argument("--refresh", action="store_true", help="Refresh 5m bars first")
    parser.add_argument("--no-gate", action="store_true", help="Skip paper gate messaging")
    args = parser.parse_args()

    settings = load_settings()
    cfg = load_spy_day_config(settings)
    bars = get_spy_5m(settings, refresh=args.refresh, rth_only=True)
    if bars.empty:
        raise SystemExit("No 5m bars — run scripts/refresh_spy_bars.py first")

    result = run_spy_day_backtest(bars, cfg=cfg)
    m = result.metrics
    report_dir = ROOT / "data" / "journal" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    md_path = report_dir / f"spy_day_backtest_{stamp}.md"
    json_path = report_dir / f"spy_day_backtest_{stamp}.json"

    gate_ok = (
        m.get("n_trades", 0) >= cfg.backtest_gate_min_trades
        and (
            m.get("profit_factor", 0) == float("inf")
            or m.get("profit_factor", 0) >= cfg.backtest_gate_pf
        )
    )

    lines = [
        f"# SPY Day 0DTE Backtest — {stamp}",
        "",
        f"Bars: **{len(bars)}** ({bars.index.min()} → {bars.index.max()})",
        f"Patterns: `{', '.join(cfg.patterns)}`",
        f"Score gate: **{cfg.score_threshold}** | TP **{cfg.profit_target_pct:.0%}** / SL **{cfg.stop_loss_pct:.0%}**",
        "",
        "## Metrics",
        "",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| Trades | {int(m.get('n_trades', 0))} |",
        f"| Win rate | {m.get('win_rate', 0):.1%} |",
        f"| Expectancy $/trade | ${m.get('expectancy', 0):.2f} |",
        f"| Profit factor | {m.get('profit_factor', 0):.2f} |",
        f"| Max drawdown | {m.get('max_drawdown', 0):.1%} |",
        f"| Total return | {m.get('total_return', 0):.1%} |",
        f"| Trades/day | {m.get('trades_per_day', 0):.2f} |",
        "",
        f"Paper gate (PF≥{cfg.backtest_gate_pf}, n≥{cfg.backtest_gate_min_trades}): "
        f"**{'PASS' if gate_ok else 'FAIL'}**",
        "",
        "## By pattern",
        "",
    ]
    for pat, stats in sorted(result.by_pattern.items()):
        lines.append(
            f"- **{pat}**: n={int(stats['n_trades'])} WR={stats['win_rate']:.1%} "
            f"E=${stats['expectancy']:.2f} PF={stats['profit_factor']:.2f} "
            f"PnL=${stats['total_pnl']:.0f}"
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "metrics": m,
        "by_pattern": result.by_pattern,
        "gate_ok": gate_ok,
        "n_bars": len(bars),
        "config": {
            "score_threshold": cfg.score_threshold,
            "profit_target_pct": cfg.profit_target_pct,
            "stop_loss_pct": cfg.stop_loss_pct,
            "patterns": cfg.patterns,
        },
    }
    # JSON-safe inf
    def _san(o):
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items()}
        if isinstance(o, float) and o == float("inf"):
            return None
        return o

    json_path.write_text(json.dumps(_san(payload), indent=2), encoding="utf-8")
    # Latest pointer for grade/scan
    (ROOT / "artifacts" / "spy_day_backtest_latest.json").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts" / "spy_day_backtest_latest.json").write_text(
        json.dumps(_san(payload), indent=2), encoding="utf-8"
    )

    print(json.dumps(_san({"metrics": m, "gate_ok": gate_ok, "report": str(md_path)}), indent=2))
    if not args.no_gate and not gate_ok:
        print("WARNING: backtest gate FAIL — keep paper dry_run / review patterns before sizing up")


if __name__ == "__main__":
    main()
