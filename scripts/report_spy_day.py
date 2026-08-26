"""Append SPY-day lane snapshot into today's journal report."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from stockpro.config import ROOT, load_settings
from stockpro.journal import Journal
from stockpro.spy_day.session import load_spy_day_config


def main() -> None:
    settings = load_settings()
    cfg = load_spy_day_config(settings)
    today = date.today().isoformat()
    report_path = ROOT / "data" / "journal" / "reports" / f"{today}.md"
    bt_path = ROOT / "artifacts" / "spy_day_backtest_latest.json"

    journal = Journal(settings)
    trades = journal.load_trades()
    decisions = journal.load_decisions() if hasattr(journal, "load_decisions") else None

    spy_trades = 0
    spy_pnl = 0.0
    if trades is not None and len(trades):
        t = trades.copy()
        if "ticker" in t.columns:
            mask = t["ticker"].astype(str).str.upper() == cfg.symbol
            if "timestamp" in t.columns:
                mask &= t["timestamp"].astype(str).str.startswith(today)
            spy = t.loc[mask]
            spy_trades = len(spy)
            if "pnl" in spy.columns:
                spy_pnl = float(spy["pnl"].fillna(0).sum())

    bt = {}
    if bt_path.exists():
        bt = json.loads(bt_path.read_text(encoding="utf-8"))

    section = [
        "",
        "## SPY Day (0DTE / 5m)",
        "",
        f"- Enabled: **{cfg.enabled}** | score≥{cfg.score_threshold} | TP {cfg.profit_target_pct:.0%} / SL {cfg.stop_loss_pct:.0%}",
        f"- Today journal trades (SPY): **{spy_trades}** | journal PnL sum: **${spy_pnl:.2f}**",
        f"- Backtest gate: **{'PASS' if bt.get('gate_ok') else 'FAIL / missing'}**",
    ]
    if bt.get("metrics"):
        m = bt["metrics"]
        section.append(
            f"- Backtest baseline: n={int(m.get('n_trades', 0))} "
            f"WR={m.get('win_rate', 0):.1%} E=${m.get('expectancy', 0):.2f} "
            f"PF={m.get('profit_factor', 0):.2f} DD={m.get('max_drawdown', 0):.1%}"
        )
    section.append(f"- Generated {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}")
    section.append("")

    text = "\n".join(section)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        existing = report_path.read_text(encoding="utf-8")
        if "## SPY Day" in existing:
            # replace prior section
            pre = existing.split("## SPY Day")[0].rstrip()
            report_path.write_text(pre + "\n" + text, encoding="utf-8")
        else:
            report_path.write_text(existing.rstrip() + "\n" + text, encoding="utf-8")
    else:
        report_path.write_text("# Daily report\n" + text, encoding="utf-8")

    out = ROOT / "artifacts" / "spy_day_daily.json"
    out.write_text(
        json.dumps(
            {
                "date": today,
                "spy_trades": spy_trades,
                "spy_pnl": spy_pnl,
                "gate_ok": bt.get("gate_ok"),
                "backtest_metrics": bt.get("metrics"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"report": str(report_path), "spy_trades": spy_trades, "gate_ok": bt.get("gate_ok")}, indent=2))


if __name__ == "__main__":
    main()
