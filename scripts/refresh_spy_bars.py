"""Refresh SPY 5m bars from Alpaca into parquet cache (RTH + extended/premarket)."""

from __future__ import annotations

import argparse

from stockpro.config import load_settings
from stockpro.data import filter_premarket, load_cached_bars_5m_ext, refresh_bars_5m


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh SPY 5m Alpaca bars (extended hours + RTH caches)"
    )
    parser.add_argument("--full", action="store_true", help="Force full history re-download")
    parser.add_argument("--days", type=int, default=None, help="Override history_days")
    args = parser.parse_args()
    settings = load_settings()
    df = refresh_bars_5m(settings, history_days=args.days, force_full=args.full)
    ext = load_cached_bars_5m_ext(settings)
    pm = filter_premarket(ext)
    print(f"OK RTH: {len(df)} bars")
    print(f"OK EXT: {len(ext)} bars | premarket: {len(pm)} bars")


if __name__ == "__main__":
    main()
