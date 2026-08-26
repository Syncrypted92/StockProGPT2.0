"""Daily performance grading from Alpaca + local journal + model accuracy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from stockpro.accuracy import resolve_predictions
from stockpro.broker import AlpacaBroker
from stockpro.config import ROOT, Settings
from stockpro.journal import Journal
from stockpro.risk import underlying_from_option_symbol


@dataclass
class DayGrade:
    date: str
    equity: float
    cash: float
    open_positions: int
    unrealized_pl: float
    day_pnl: float
    starting_equity: float
    drawdown_pct: float
    n_decisions: int
    n_orders: int
    n_exits: int
    closed_trades: int
    win_rate: float | None
    profit_factor: float | None
    expectancy: float | None
    model_n_resolved: int
    model_accuracy_3class: float | None
    model_accuracy_directional: float | None
    model_accuracy_high_conf: float | None
    model_resolved_today: int
    model_grade: str
    ops_grade: str
    risk_grade: str
    edge_grade: str
    overall_grade: str
    verdict: str
    notes: str = ""


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _letter_from_scores(*grades: str) -> str:
    rank = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1, "N": 3}
    vals = [rank.get(g, 3) for g in grades]
    avg = sum(vals) / len(vals)
    if avg >= 4.5:
        return "A"
    if avg >= 3.5:
        return "B"
    if avg >= 2.5:
        return "C"
    if avg >= 1.5:
        return "D"
    return "F"


def _closed_exits(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "pnl" not in trades.columns:
        return pd.DataFrame()
    closed = trades.copy()
    if "side" in closed.columns:
        exits = closed[closed["side"].astype(str).str.contains("close|sell", case=False, na=False)]
        if not exits.empty:
            closed = exits
    closed = closed[pd.to_numeric(closed["pnl"], errors="coerce").fillna(0) != 0]
    return closed


def _rolling_trade_stats(trades: pd.DataFrame) -> dict[str, float | None]:
    closed = _closed_exits(trades)
    if closed.empty:
        return {"closed_trades": 0, "win_rate": None, "profit_factor": None, "expectancy": None}

    pnls = pd.to_numeric(closed["pnl"], errors="coerce").fillna(0.0)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(losses.sum()) if len(losses) else 0.0
    pf = (gross_win / abs(gross_loss)) if gross_loss < 0 else (float("inf") if gross_win > 0 else None)
    return {
        "closed_trades": int(len(pnls)),
        "win_rate": float((pnls > 0).mean()),
        "profit_factor": pf,
        "expectancy": float(pnls.mean()),
    }


def _edge_breakdown(trades: pd.DataFrame, acc_by_ticker: dict[str, float] | None = None) -> str:
    """Markdown: hit rate by ticker, hold time, stop vs target mix, hot stop-rate names."""
    closed = _closed_exits(trades)
    lines = ["## Edge breakdown (closed trades)", ""]
    if closed.empty:
        lines.append("_No closed trades yet._")
        return "\n".join(lines)

    df = closed.copy()
    df["_root"] = df["ticker"].astype(str).map(
        lambda t: underlying_from_option_symbol(t) if len(str(t)) > 6 else str(t).upper()
    )
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
    reasons = df["exit_reason"].astype(str).str.lower() if "exit_reason" in df.columns else pd.Series([""] * len(df))
    n_stop = int((reasons == "stop_loss").sum())
    n_tgt = int((reasons == "profit_target").sum())
    n_other = int(len(df) - n_stop - n_tgt)
    lines.append(f"- Exit mix: **{n_tgt}** profit target / **{n_stop}** stop / **{n_other}** other")

    # Avg hold time when we can join entry→exit on contract
    hold_note = "_n/a_"
    if "timestamp" in df.columns and "contract" in trades.columns:
        all_t = trades.copy()
        all_t["ts"] = pd.to_datetime(all_t["timestamp"], utc=True, errors="coerce")
        holds: list[float] = []
        for _, ex in df.iterrows():
            contract = str(ex.get("contract") or "")
            if not contract or pd.isna(ex.get("timestamp")):
                continue
            ex_ts = pd.to_datetime(ex["timestamp"], utc=True, errors="coerce")
            ents = all_t[
                (all_t["contract"].astype(str) == contract)
                & (all_t["side"].astype(str).str.contains("open|buy", case=False, na=False))
                & (all_t["ts"] <= ex_ts)
            ]
            if ents.empty:
                continue
            ent_ts = ents["ts"].max()
            if pd.notna(ent_ts) and pd.notna(ex_ts):
                holds.append((ex_ts - ent_ts).total_seconds() / 3600.0)
        if holds:
            hold_note = f"**{sum(holds) / len(holds):.1f}h** avg ({len(holds)} matched)"
    lines.append(f"- Avg hold time: {hold_note}")
    lines.append("")
    lines.append("| Ticker | Closed | Win rate | Expectancy | Stops | Model hit |")
    lines.append("|--------|-------:|---------:|-----------:|------:|----------:|")
    acc_by_ticker = acc_by_ticker or {}
    for root, g in df.groupby("_root"):
        wr = float((g["pnl"] > 0).mean())
        exp = float(g["pnl"].mean())
        stops = int((g["exit_reason"].astype(str).str.lower() == "stop_loss").sum()) if "exit_reason" in g.columns else 0
        hit = acc_by_ticker.get(str(root))
        hit_s = f"{hit:.0%}" if hit is not None else "n/a"
        flag = " ⚠" if len(g) >= 3 and stops / len(g) >= 0.70 else ""
        lines.append(
            f"| {root}{flag} | {len(g)} | {wr:.0%} | ${exp:.2f} | {stops}/{len(g)} | {hit_s} |"
        )
    lines.append("")
    lines.append("⚠ = stop-rate ≥ 70% over closed sample for that ticker.")
    return "\n".join(lines)


def _model_grade(acc_stats: dict[str, Any]) -> str:
    n = int(acc_stats.get("n_resolved") or 0)
    dir_acc = acc_stats.get("accuracy_directional")
    high = acc_stats.get("accuracy_high_confidence")
    if n < 20:
        return "N"
    # Prefer high-confidence accuracy when available
    score = high if high is not None else dir_acc
    if score is None:
        return "N"
    if score >= 0.55:
        return "A"
    if score >= 0.52:
        return "B"
    if score >= 0.50:
        return "C"
    if score >= 0.47:
        return "D"
    return "F"


def grade_day(settings: Settings, as_of: date | None = None, notes: str = "") -> DayGrade:
    as_of = as_of or date.today()
    gates = settings.get("paper_gates", default={}) or {}
    max_dd = float(gates.get("max_drawdown_pct", 0.15))
    soft_dd = float(gates.get("soft_drawdown_pct", 0.10))
    min_days = int(gates.get("min_trading_days", 20))
    start_equity = float(
        (settings.get("backtest", default={}) or {}).get("initial_equity", 100000.0)
    )

    # Resolve due predictions before grading
    acc_stats = resolve_predictions(settings, as_of=as_of)
    m_grade = _model_grade(acc_stats)

    broker = AlpacaBroker(settings, dry_run=False)
    broker.connect()
    account = broker.get_account()
    positions = broker.list_positions()
    equity = _safe_float(account.get("equity"), start_equity)
    cash = _safe_float(account.get("cash"))
    unrealized = sum(_safe_float(p.get("unrealized_pl")) for p in positions)

    journal = Journal(settings)
    decisions = journal.load_decisions()
    trades = journal.load_trades()

    day_str = as_of.isoformat()
    n_decisions = n_orders = n_exits = 0
    if not decisions.empty and "timestamp" in decisions.columns:
        ts = pd.to_datetime(decisions["timestamp"], utc=True, errors="coerce")
        day_mask = ts.dt.date == as_of
        day_dec = decisions.loc[day_mask]
        n_decisions = int(len(day_dec))
        if "action" in day_dec.columns:
            n_orders = int((day_dec["action"] == "order").sum())
            n_exits = int((day_dec["action"] == "exit").sum())

    day_pnl = 0.0
    if journal.daily_pnl_path.exists():
        dp = pd.read_csv(journal.daily_pnl_path)
        if not dp.empty and "date" in dp.columns:
            hit = dp[dp["date"].astype(str) == day_str]
            if not hit.empty:
                day_pnl = _safe_float(hit.iloc[-1].get("pnl"))

    drawdown_pct = 0.0
    if start_equity > 0:
        drawdown_pct = min(0.0, equity / start_equity - 1.0)
    grades_path = Path(journal.dir) / "daily_grades.csv"
    peak = start_equity
    if grades_path.exists():
        hist = pd.read_csv(grades_path)
        if not hist.empty and "equity" in hist.columns:
            peak = max(peak, float(hist["equity"].max()), equity)
            drawdown_pct = min(0.0, equity / peak - 1.0)

    stats = _rolling_trade_stats(trades)

    if account.get("mock"):
        ops = "F"
        ops_note = "mock account / not connected"
    elif n_decisions == 0 and n_orders == 0:
        ops = "C"
        ops_note = "no scan activity logged today"
    else:
        ops = "A"
        ops_note = "connected; journal activity present"

    dd_abs = abs(drawdown_pct)
    if dd_abs > max_dd:
        risk = "F"
    elif dd_abs > soft_dd:
        risk = "D"
    elif day_pnl <= -float((settings.get("risk", default={}) or {}).get("max_daily_loss_pct", 0.02)) * equity:
        risk = "C"
    else:
        risk = "A"

    closed = int(stats["closed_trades"] or 0)
    pf = stats["profit_factor"]
    wr = stats["win_rate"]
    exp = stats["expectancy"]
    if closed < 5:
        edge = "N"
        edge_note = f"only {closed} closed trades — too early"
    else:
        edge_note = ""
        pf_v = float(pf) if pf is not None and pf != float("inf") else (99.0 if pf == float("inf") else 0.0)
        if pf_v >= 1.3 and (wr or 0) >= 0.45 and (exp or 0) > 0:
            edge = "A"
        elif pf_v >= 1.0 and (exp or 0) >= 0:
            edge = "B"
        elif pf_v >= 0.9:
            edge = "C"
        elif pf_v >= 0.7:
            edge = "D"
        else:
            edge = "F"

    overall = _letter_from_scores(ops, risk, edge, m_grade)
    if overall in {"A", "B"} and closed >= min_days // 2:
        verdict = "On track vs paper gates — keep going."
    elif edge == "N" or m_grade == "N":
        verdict = "Collecting sample size — track PnL and model accuracy together."
    elif overall in {"D", "F"}:
        verdict = "Below target — review signals/exits before adding size."
    else:
        verdict = "Mixed — continue paper; do not go live."

    note_parts = [ops_note, edge_note, notes]
    return DayGrade(
        date=day_str,
        equity=equity,
        cash=cash,
        open_positions=len(positions),
        unrealized_pl=unrealized,
        day_pnl=day_pnl,
        starting_equity=start_equity,
        drawdown_pct=drawdown_pct,
        n_decisions=n_decisions,
        n_orders=n_orders,
        n_exits=n_exits,
        closed_trades=closed,
        win_rate=wr,
        profit_factor=(None if pf == float("inf") else pf),
        expectancy=exp,
        model_n_resolved=int(acc_stats.get("n_resolved") or 0),
        model_accuracy_3class=acc_stats.get("accuracy_3class"),
        model_accuracy_directional=acc_stats.get("accuracy_directional"),
        model_accuracy_high_conf=acc_stats.get("accuracy_high_confidence"),
        model_resolved_today=int(acc_stats.get("resolved_today") or 0),
        model_grade=m_grade,
        ops_grade=ops,
        risk_grade=risk,
        edge_grade=edge,
        overall_grade=overall,
        verdict=verdict,
        notes="; ".join(p for p in note_parts if p),
    )


def persist_grade(settings: Settings, grade: DayGrade) -> tuple[Path, Path]:
    journal = Journal(settings)
    grades_path = Path(journal.dir) / "daily_grades.csv"
    reports_dir = Path(journal.dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    row = asdict(grade)
    df = pd.DataFrame([row])
    if grades_path.exists():
        prev = pd.read_csv(grades_path)
        prev = prev[prev["date"].astype(str) != grade.date]
        df = pd.concat([prev, df], ignore_index=True)
        df.to_csv(grades_path, index=False)
    else:
        df.to_csv(grades_path, index=False)

    report_path = reports_dir / f"{grade.date}.md"
    pf = grade.profit_factor
    pf_s = f"{pf:.2f}" if isinstance(pf, (int, float)) and pf is not None else "n/a"
    wr_s = f"{grade.win_rate:.1%}" if grade.win_rate is not None else "n/a"
    exp_s = f"${grade.expectancy:.2f}" if grade.expectancy is not None else "n/a"
    a3 = f"{grade.model_accuracy_3class:.1%}" if grade.model_accuracy_3class is not None else "n/a"
    ad = f"{grade.model_accuracy_directional:.1%}" if grade.model_accuracy_directional is not None else "n/a"
    ah = f"{grade.model_accuracy_high_conf:.1%}" if grade.model_accuracy_high_conf is not None else "n/a"

    trades = journal.load_trades()
    acc_stats = resolve_predictions(settings, as_of=date.fromisoformat(grade.date))
    edge_md = _edge_breakdown(trades, acc_stats.get("accuracy_by_ticker") or {})

    md = f"""# Daily Grade — {grade.date}

**Overall: {grade.overall_grade}** — {grade.verdict}

| Lens | Grade |
|------|-------|
| Ops | {grade.ops_grade} |
| Risk | {grade.risk_grade} |
| Edge (PnL) | {grade.edge_grade} |
| Model accuracy | {grade.model_grade} |

## Account (Alpaca paper)

| Field | Value |
|-------|------:|
| Equity | ${grade.equity:,.2f} |
| Cash | ${grade.cash:,.2f} |
| Open positions | {grade.open_positions} |
| Unrealized P/L | ${grade.unrealized_pl:,.2f} |
| Drawdown vs peak | {grade.drawdown_pct:.2%} |
| Day P/L (journal) | ${grade.day_pnl:,.2f} |

## Activity today

- Decisions logged: **{grade.n_decisions}**
- Entry orders: **{grade.n_orders}**
- Exits: **{grade.n_exits}**

## Model accuracy (realized forward returns)

| Metric | Value |
|--------|------:|
| Predictions resolved (all-time) | {grade.model_n_resolved} |
| Resolved today | {grade.model_resolved_today} |
| 3-class accuracy | {a3} |
| Directional accuracy (non-flat preds) | {ad} |
| High-confidence accuracy | {ah} |

Predictions live in `data/journal/predictions.csv` and resolve after the model horizon (default 5 sessions).

## Rolling edge (closed trades)

| Metric | Value |
|--------|------:|
| Closed trades | {grade.closed_trades} |
| Win rate | {wr_s} |
| Profit factor | {pf_s} |
| Expectancy $/trade | {exp_s} |

{edge_md}

## Notes

{grade.notes or "_None_"}

---
Generated {datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
See also: [PAPER_TRADING_PLAN.md](../../../docs/PAPER_TRADING_PLAN.md) · [Week baseline](../../../docs/WEEK_2026-07-13_BASELINE.md)
"""
    report_path.write_text(md, encoding="utf-8")
    return grades_path, report_path


def summary_table(settings: Settings) -> str:
    path = Path((settings.get("journal", default={}) or {}).get("directory", "data/journal")) / "daily_grades.csv"
    path = ROOT / path if not path.is_absolute() else path
    if not path.exists():
        return "No daily_grades.csv yet — run scripts/daily_grade.py after a session."
    df = pd.read_csv(path)
    if df.empty:
        return "daily_grades.csv is empty."
    cols = [
        c
        for c in [
            "date",
            "overall_grade",
            "model_grade",
            "model_accuracy_directional",
            "equity",
            "closed_trades",
            "profit_factor",
            "verdict",
        ]
        if c in df.columns
    ]
    return df[cols].to_string(index=False)
