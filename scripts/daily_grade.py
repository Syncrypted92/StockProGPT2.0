"""End-of-day paper performance grade (Alpaca + journal)."""

from __future__ import annotations

import argparse
import json
from datetime import date

from stockpro.config import load_settings
from stockpro.grading import grade_day, persist_grade, summary_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade today's paper trading performance")
    parser.add_argument("--date", help="YYYY-MM-DD (default: today)")
    parser.add_argument("--notes", default="", help="Optional note stored on the grade")
    parser.add_argument("--summary", action="store_true", help="Print rolling grades table only")
    args = parser.parse_args()

    settings = load_settings()
    if args.summary:
        print(summary_table(settings))
        return

    as_of = date.fromisoformat(args.date) if args.date else date.today()
    grade = grade_day(settings, as_of=as_of, notes=args.notes)
    grades_path, report_path = persist_grade(settings, grade)

    print(json.dumps({
        "date": grade.date,
        "overall_grade": grade.overall_grade,
        "ops_grade": grade.ops_grade,
        "risk_grade": grade.risk_grade,
        "edge_grade": grade.edge_grade,
        "model_grade": grade.model_grade,
        "model_accuracy_3class": grade.model_accuracy_3class,
        "model_accuracy_directional": grade.model_accuracy_directional,
        "model_accuracy_high_conf": grade.model_accuracy_high_conf,
        "model_n_resolved": grade.model_n_resolved,
        "equity": grade.equity,
        "drawdown_pct": grade.drawdown_pct,
        "closed_trades": grade.closed_trades,
        "win_rate": grade.win_rate,
        "profit_factor": grade.profit_factor,
        "expectancy": grade.expectancy,
        "verdict": grade.verdict,
        "grades_csv": str(grades_path),
        "report_md": str(report_path),
    }, indent=2))
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
