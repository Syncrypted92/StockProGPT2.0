"""Paper trading and model evaluation reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from stockpro.accuracy import resolve_predictions
from stockpro.backtest import run_walk_forward_backtest
from stockpro.config import ROOT, Settings
from stockpro.data import download_universe
from stockpro.journal import Journal
from stockpro.risk import underlying_from_option_symbol


@dataclass
class EvalResult:
    date_from: str
    date_to: str
    report_path: Path
    summary: dict[str, Any]


def _df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _is_resolved(val: Any) -> bool:
    return str(val).strip().lower() in {"true", "1", "yes"}


def _closed_exits(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    closed = trades.copy()
    if "side" in closed.columns:
        exits = closed[closed["side"].astype(str).str.contains("close|sell", case=False, na=False)]
        if not exits.empty:
            closed = exits
    if "pnl" in closed.columns:
        closed = closed[pd.to_numeric(closed["pnl"], errors="coerce").fillna(0) != 0]
    return closed


def _predictions_in_range(settings: Settings, d_from: date, d_to: date) -> pd.DataFrame:
    path = Path((settings.get("journal", default={}) or {}).get("directory", "data/journal")) / "predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["_as_of"] = pd.to_datetime(df["as_of"], errors="coerce").dt.date
    return df[(df["_as_of"] >= d_from) & (df["_as_of"] <= d_to)].copy()


def _calibration_table(preds: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if preds.empty or not preds["resolved"].map(_is_resolved).any():
        return pd.DataFrame()
    done = preds[preds["resolved"].map(_is_resolved)].copy()
    done["confidence"] = pd.to_numeric(done["confidence"], errors="coerce")
    done["correct"] = pd.to_numeric(done["correct"], errors="coerce")
    bins = [0.0, 0.55, 0.60, 0.65, 1.01]
    labels = ["<0.55", "0.55-0.60", "0.60-0.65", "0.65+"]
    done["bucket"] = pd.cut(done["confidence"], bins=bins, labels=labels, right=False)
    rows = []
    for label in labels:
        g = done[done["bucket"] == label]
        if g.empty:
            continue
        rows.append(
            {
                "bucket": str(label),
                "n": len(g),
                "hit_rate": float(g["correct"].mean()),
                "above_threshold": str(label) in {"0.65+", "0.60-0.65"} and float(label.split("-")[0].replace("+", "")) >= threshold
                if False
                else (str(label) == "0.65+"),
            }
        )
    return pd.DataFrame(rows)


def _ticker_accuracy_vs_pnl(preds: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    closed = _closed_exits(trades)
    tickers = set()
    if not preds.empty:
        tickers.update(preds["ticker"].astype(str).unique())
    if not closed.empty:
        closed = closed.copy()
        closed["_root"] = closed["ticker"].astype(str).map(
            lambda t: underlying_from_option_symbol(t) if len(str(t)) > 6 else str(t).upper()
        )
        tickers.update(closed["_root"].unique())
    for tkr in sorted(tickers):
        p = preds[preds["ticker"].astype(str) == tkr] if not preds.empty else pd.DataFrame()
        resolved = p[p["resolved"].map(_is_resolved)] if not p.empty else pd.DataFrame()
        acc = float(pd.to_numeric(resolved["correct"], errors="coerce").mean()) if len(resolved) else None
        ct = closed[closed["_root"] == tkr] if not closed.empty and "_root" in closed.columns else pd.DataFrame()
        pnl = float(pd.to_numeric(ct["pnl"], errors="coerce").sum()) if len(ct) else 0.0
        n_trades = len(ct)
        rows.append({"ticker": tkr, "n_preds_resolved": len(resolved), "accuracy": acc, "closed_trades": n_trades, "pnl": pnl})
    return pd.DataFrame(rows)


def _trade_attribution(trades: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    closed = _closed_exits(trades)
    if closed.empty:
        return pd.DataFrame()
    out = closed.copy()
    out["_root"] = out["ticker"].astype(str).map(
        lambda t: underlying_from_option_symbol(t) if len(str(t)) > 6 else str(t).upper()
    )
    out["pnl"] = pd.to_numeric(out["pnl"], errors="coerce").fillna(0)
    if "contract" in out.columns and not decisions.empty and "contract" in decisions.columns:
        orders = decisions[decisions["action"].astype(str) == "order"].copy()
        orders["confidence"] = pd.to_numeric(orders.get("confidence"), errors="coerce")
        merged = out.merge(
            orders[["contract", "confidence", "signal"]].drop_duplicates("contract"),
            on="contract",
            how="left",
            suffixes=("", "_entry"),
        )
        out = merged
    out["model_mismatch"] = ""
    if "exit_reason" in out.columns and "signal" in out.columns:
        # Flag stop when we had directional entry — options structure issue
        stops = out["exit_reason"].astype(str).str.lower() == "stop_loss"
        out.loc[stops, "model_mismatch"] = "option_stop_despite_entry"
    return out


def _skip_analysis(decisions: pd.DataFrame, preds: pd.DataFrame, d_from: date, d_to: date) -> dict[str, Any]:
    if decisions.empty:
        return {"skip_counts": {}, "hypothetical": []}
    dec = decisions.copy()
    dec["ts"] = pd.to_datetime(dec["timestamp"], utc=True, errors="coerce")
    dec = dec[(dec["ts"].dt.date >= d_from) & (dec["ts"].dt.date <= d_to)]
    skips = dec[dec["action"].astype(str) == "skip"]
    counts = skips["reason"].astype(str).value_counts().to_dict() if "reason" in skips.columns else {}

    hypo: list[dict[str, Any]] = []
    if not preds.empty and not skips.empty:
        resolved = preds[preds["resolved"].map(_is_resolved)].copy()
        if not resolved.empty:
            resolved["correct"] = pd.to_numeric(resolved["correct"], errors="coerce")
            for _, row in skips.iterrows():
                tkr = str(row.get("ticker", ""))
                hit = resolved[resolved["ticker"].astype(str) == tkr]
                if hit.empty:
                    continue
                last = hit.sort_values("as_of").iloc[-1]
                hypo.append(
                    {
                        "ticker": tkr,
                        "skip_reason": str(row.get("reason", "")),
                        "would_have_been_correct": bool(last.get("correct")),
                        "confidence": float(last.get("confidence") or 0),
                    }
                )
    return {"skip_counts": counts, "hypothetical": hypo}


def _walk_forward_baseline(settings: Settings) -> dict[str, float]:
    uni = settings.get("universe", default={}) or {}
    bt = settings.get("backtest", default={}) or {}
    feat = settings.get("features", default={}) or {}
    model_cfg = settings.get("model", default={}) or {}
    risk = settings.get("risk", default={}) or {}
    tickers = list(uni.get("tickers", ["SPY"]))[:4]
    train_days = int(bt.get("walk_forward_train_days", 180))
    test_days = int(bt.get("walk_forward_test_days", 30))
    # Only pull enough history for walk-forward (avoid full 2000-start download)
    lookback_days = int((train_days + test_days * 3) * 1.5)
    try:
        frames = download_universe(tickers, lookback_days=lookback_days, start=None)
        if not frames:
            return {}
        result = run_walk_forward_backtest(
            frames,
            initial_equity=float(bt.get("initial_equity", 100000)),
            spread_penalty_pct=float(bt.get("spread_penalty_pct", 0.04)),
            option_premium_pct_of_spot=float(bt.get("option_premium_pct_of_spot", 0.005)),
            train_days=int(bt.get("walk_forward_train_days", 180)),
            test_days=int(bt.get("walk_forward_test_days", 30)),
            horizon=int(feat.get("forward_horizon_days", 5)),
            flat_threshold=float(feat.get("flat_threshold", 0.005)),
            probability_threshold=float(model_cfg.get("probability_threshold", 0.65)),
            max_risk_per_trade_pct=float(risk.get("max_risk_per_trade_pct", 0.01)),
            max_notional_per_trade=float(risk.get("max_notional_per_trade", 250)),
            max_contracts=int(risk.get("max_contracts_per_trade", 1)),
        )
        return result.metrics
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _load_model_meta(settings: Settings) -> dict[str, Any]:
    model_cfg = settings.get("model", default={}) or {}
    meta_path = ROOT / model_cfg.get("artifact_dir", "artifacts") / model_cfg.get("meta_filename", "model_meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _recommendations(
    ticker_table: pd.DataFrame,
    acc_stats: dict[str, Any],
    wf: dict[str, float],
) -> list[str]:
    recs: list[str] = []
    acc = acc_stats.get("accuracy_3class")
    if acc is not None and acc < 0.52:
        recs.append("Model accuracy is near coin-flip — do not increase size until edge improves.")
    if not ticker_table.empty:
        bad = ticker_table[(ticker_table["closed_trades"] >= 2) & (ticker_table["pnl"] < -50)]
        for _, r in bad.iterrows():
            recs.append(f"Review universe: {r['ticker']} lost ${abs(r['pnl']):.0f} with {int(r['closed_trades'])} closed trades.")
    if wf.get("profit_factor") is not None and wf["profit_factor"] < 1.0:
        recs.append("Walk-forward profit factor < 1.0 — technical-only baseline is not yet tradable.")
    if not recs:
        recs.append("Continue paper with current rules; re-evaluate after more resolved predictions.")
    return recs


def run_evaluation(
    settings: Settings,
    *,
    date_from: date,
    date_to: date,
    run_backtest: bool = True,
) -> EvalResult:
    journal = Journal(settings)
    resolve_predictions(settings, as_of=date_to)
    preds = _predictions_in_range(settings, date_from, date_to)
    trades = journal.load_trades()
    decisions = journal.load_decisions()

    model_cfg = settings.get("model", default={}) or {}
    threshold = float(model_cfg.get("probability_threshold", 0.65))
    acc_stats = resolve_predictions(settings, as_of=date_to)
    cal = _calibration_table(preds, threshold)
    ticker_table = _ticker_accuracy_vs_pnl(preds, trades)
    attribution = _trade_attribution(trades, decisions)
    skip_info = _skip_analysis(decisions, preds, date_from, date_to)
    wf = _walk_forward_baseline(settings) if run_backtest else {}
    meta = _load_model_meta(settings)
    recs = _recommendations(ticker_table, acc_stats, wf)

    closed = _closed_exits(trades)
    pnls = pd.to_numeric(closed["pnl"], errors="coerce").fillna(0) if not closed.empty else pd.Series(dtype=float)
    paper_stats = {
        "closed_trades": int(len(closed)),
        "total_pnl": float(pnls.sum()) if len(pnls) else 0.0,
        "win_rate": float((pnls > 0).mean()) if len(pnls) else None,
        "expectancy": float(pnls.mean()) if len(pnls) else None,
    }

    reports_dir = journal.dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_name = f"eval_{date_to.isoformat()}.md"
    report_path = reports_dir / report_name

    def _pct(v: Any) -> str:
        return f"{float(v):.1%}" if v is not None and pd.notna(v) else "n/a"

    cal_md = _df_to_md(cal) if not cal.empty else "_No resolved predictions in range._"
    ticker_md = _df_to_md(ticker_table) if not ticker_table.empty else "_No ticker data._"

    attr_lines = []
    if not attribution.empty:
        for _, r in attribution.iterrows():
            root = r.get("_root", r.get("ticker", ""))
            attr_lines.append(
                f"| {root} | {r.get('contract', '')} | {r.get('signal', '')} | "
                f"{r.get('confidence', 'n/a')} | {r.get('exit_reason', '')} | ${float(r.get('pnl', 0)):.2f} |"
            )
    attr_md = (
        "| Ticker | Contract | Signal | Conf | Exit | PnL |\n|--------|----------|--------|------|------|-----|\n"
        + "\n".join(attr_lines)
        if attr_lines
        else "_No closed trades._"
    )

    skip_counts = skip_info.get("skip_counts") or {}
    skip_md = "\n".join(f"- **{k}**: {v}" for k, v in sorted(skip_counts.items())) or "_None_"
    hypo = skip_info.get("hypothetical") or []
    hypo_md = "\n".join(
        f"- {h['ticker']} ({h['skip_reason']}): would_be_correct={h['would_have_been_correct']}, conf={h['confidence']:.2f}"
        for h in hypo[:20]
    ) or "_No skip hypotheticals (need resolved predictions)._"

    wf_md = "\n".join(f"- **{k}**: {v}" for k, v in wf.items()) if wf else "_Backtest skipped or failed._"
    meta_train = meta.get("test_accuracy")
    meta_hit = meta.get("hit_rate_directional")

    md = f"""# Evaluation Report — {date_from.isoformat()} to {date_to.isoformat()}

Generated {datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}

## Executive summary

- Paper closed PnL: **${paper_stats['total_pnl']:.2f}** ({paper_stats['closed_trades']} trades)
- Realized model accuracy (all-time resolved): **{_pct(acc_stats.get('accuracy_3class'))}** (n={acc_stats.get('n_resolved', 0)})
- High-confidence accuracy: **{_pct(acc_stats.get('accuracy_high_confidence'))}**
- Train-time test accuracy (artifact): **{_pct(meta_train)}** | directional: **{_pct(meta_hit)}**

## Signal quality — calibration

{cal_md}

## Per-ticker: model accuracy vs PnL

{ticker_md}

## Trade attribution

{attr_md}

## Skip analysis

{skip_md}

### Hypothetical outcomes for skipped tickers

{hypo_md}

## Walk-forward baseline (technical-only)

{wf_md}

## Recommendations

{chr(10).join(f'- {r}' for r in recs)}

---
See also: [WEEK_2026-07-13_BASELINE.md](../../../docs/WEEK_2026-07-13_BASELINE.md)
"""
    report_path.write_text(md, encoding="utf-8")

    summary = {
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "paper_stats": paper_stats,
        "acc_stats": {k: v for k, v in acc_stats.items() if k != "accuracy_by_ticker"},
        "walk_forward": wf,
        "recommendations": recs,
        "report_path": str(report_path),
    }
    return EvalResult(
        date_from=date_from.isoformat(),
        date_to=date_to.isoformat(),
        report_path=report_path,
        summary=summary,
    )
