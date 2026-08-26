"""Market data adapters (Yahoo daily; Alpaca 5m for spy_day)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
import yfinance as yf

from stockpro.data.bars_5m import (
    download_alpaca_bars_5m,
    filter_premarket,
    filter_rth,
    get_spy_5m,
    get_spy_premarket_5m,
    load_cached_bars_5m,
    load_cached_bars_5m_ext,
    premarket_session_stats,
    refresh_bars_5m,
    save_bars_5m,
    save_bars_5m_ext,
)

DEFAULT_START = "2000-01-01"

__all__ = [
    "DEFAULT_START",
    "download_ohlcv",
    "download_universe",
    "download_alpaca_bars_5m",
    "filter_premarket",
    "filter_rth",
    "get_spy_5m",
    "get_spy_premarket_5m",
    "load_cached_bars_5m",
    "load_cached_bars_5m_ext",
    "premarket_session_stats",
    "refresh_bars_5m",
    "save_bars_5m",
    "save_bars_5m_ext",
]


def download_ohlcv(
    ticker: str,
    lookback_days: int | None = None,
    start: str | None = DEFAULT_START,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV through the latest available session."""
    end = end or datetime.now(timezone.utc)
    end_str = end.strftime("%Y-%m-%d")

    if start:
        start_str = start
    elif lookback_days:
        start_str = (end - pd.Timedelta(days=int(lookback_days * 1.7))).strftime("%Y-%m-%d")
    else:
        start_str = DEFAULT_START

    df = yf.download(
        ticker,
        start=start_str,
        end=end_str,
        auto_adjust=False,
        progress=False,
        actions=False,
    )
    if df.empty:
        raise ValueError(f"No OHLCV data returned for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns=str.title)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker} missing columns: {missing}")

    df = df[required + (["Adj Close"] if "Adj Close" in df.columns else [])].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().dropna(subset=["Close", "Volume"])

    if lookback_days and not start and len(df) > lookback_days:
        df = df.iloc[-lookback_days:]

    df["Ticker"] = ticker
    return df


def download_universe(
    tickers: Iterable[str],
    lookback_days: int | None = None,
    start: str | None = DEFAULT_START,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            out[ticker] = download_ohlcv(ticker, lookback_days=lookback_days, start=start)
            print(
                f"[data] {ticker}: {len(out[ticker])} bars "
                f"{out[ticker].index.min().date()} -> {out[ticker].index.max().date()}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[data] skip {ticker}: {exc}")
    return out
