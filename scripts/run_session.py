"""
Run a paper-trading session phase against Alpaca.

Phases:
  morning   — smoke + swing scan --submit
  zerodte   — legacy multi-ETF 0DTE (if enabled)
  spy-day   — SPY 0DTE 5m pattern scan
  afternoon — manage positions --submit (exits; includes 0DTE / spy_day rules)
  eod       — daily grade + spy-day report snippet
  all       — morning + spy-day + afternoon + eod
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(args: list[str]) -> int:
    print(f"\n>>> {' '.join(args)}")
    completed = subprocess.run(args, cwd=str(ROOT))
    return int(completed.returncode)


def morning(submit: bool) -> int:
    code = _run([PY, "scripts/smoke_test.py"])
    if code != 0:
        return code
    scan = [PY, "scripts/scan_and_trade.py"]
    scan.append("--submit" if submit else "--dry-run")
    return _run(scan)


def zerodte(submit: bool) -> int:
    from stockpro.config import load_settings

    settings = load_settings()
    zd = settings.get("zerodte", default={}) or {}
    if not zd.get("enabled", False):
        print("[run_session] zerodte.enabled=false — skipping legacy 0DTE scan")
        return 0
    scan = [PY, "scripts/scan_and_trade.py", "--zerodte"]
    scan.append("--submit" if submit else "--dry-run")
    return _run(scan)


def spy_day(submit: bool) -> int:
    from stockpro.config import load_settings

    settings = load_settings()
    sd = settings.get("spy_day", default={}) or {}
    if not sd.get("enabled", False):
        print("[run_session] spy_day.enabled=false — skipping")
        return 0
    scan = [PY, "scripts/scan_spy_day.py", "--refresh"]
    scan.append("--submit" if submit else "--dry-run")
    return _run(scan)


def afternoon(submit: bool) -> int:
    manage = [PY, "scripts/manage_positions.py"]
    manage.append("--submit" if submit else "--dry-run")
    return _run(manage)


def eod() -> int:
    code = _run([PY, "scripts/daily_grade.py"])
    if code != 0:
        return code
    return _run([PY, "scripts/report_spy_day.py"])


def main() -> None:
    parser = argparse.ArgumentParser(description="StockPro scheduled session runner")
    parser.add_argument(
        "phase",
        choices=["morning", "zerodte", "spy-day", "afternoon", "eod", "all"],
        help="Which session phase to run",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--submit", action="store_true", help="Submit paper orders")
    args = parser.parse_args()

    submit = args.submit or (
        not args.dry_run
        and args.phase in {"morning", "zerodte", "spy-day", "afternoon", "all"}
    )
    if args.dry_run:
        submit = False

    codes = []
    if args.phase in {"morning", "all"}:
        codes.append(morning(submit))
    if args.phase in {"zerodte", "all"}:
        codes.append(zerodte(submit))
    if args.phase in {"spy-day", "all"}:
        codes.append(spy_day(submit))
    if args.phase in {"afternoon", "all"}:
        codes.append(afternoon(submit))
    if args.phase in {"eod", "all"}:
        codes.append(eod())

    bad = [c for c in codes if c != 0]
    raise SystemExit(bad[0] if bad else 0)


if __name__ == "__main__":
    main()
