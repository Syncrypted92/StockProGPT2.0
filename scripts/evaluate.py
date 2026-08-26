"""Evaluate paper trading performance and model accuracy over a date range."""

from __future__ import annotations

import argparse
import json
from datetime import date

from stockpro.config import load_settings
from stockpro.evaluation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="StockPro evaluation report")
    parser.add_argument("--from", dest="date_from", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--no-backtest", action="store_true", help="Skip walk-forward backtest")
    args = parser.parse_args()

    settings = load_settings()
    d_from = date.fromisoformat(args.date_from)
    d_to = date.fromisoformat(args.date_to)
    result = run_evaluation(
        settings,
        date_from=d_from,
        date_to=d_to,
        run_backtest=not args.no_backtest,
    )
    print(json.dumps(result.summary, indent=2, default=str))
    print(f"\nWrote {result.report_path}")


if __name__ == "__main__":
    main()
