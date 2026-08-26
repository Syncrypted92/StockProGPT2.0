"""Tiingo IEX 5m bars (optional afterHours) — research cache only, not live."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv

from stockpro.config import ROOT

ET = ZoneInfo("America/New_York")


def _tiingo_key() -> str:
    load_dotenv(ROOT / ".env")
    return (os.getenv("TIINGO_API_KEY") or os.getenv("TIINGO_TOKEN") or "").strip()


def tiingo_iex_path(symbol: str = "SPY", *, after_hours: bool = True) -> Path:
    tag = "ah" if after_hours else "rth"
    return ROOT / "data" / "bars" / f"{symbol.upper()}_5m_tiingo_{tag}.parquet"


def fetch_tiingo_iex_5m(
    symbol: str,
    start: date,
    end: date,
    *,
    after_hours: bool = True,
    token: str | None = None,
) -> pd.DataFrame:
    """Pull Tiingo IEX historical 5m. afterHours≈pre/post on IEX (often from ~08:00 ET)."""
    token = token or _tiingo_key()
    if not token:
        raise RuntimeError("TIINGO_API_KEY missing")
    url = f"https://api.tiingo.com/iex/{symbol.lower()}/prices"
    params = {
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "resampleFreq": "5min",
        "columns": "open,high,low,close,volume",
        "afterHours": "true" if after_hours else "false",
        "token": token,
    }
    r = requests.get(url, params=params, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Tiingo IEX {r.status_code}: {r.text[:300]}")
    raw = r.json()
    if not raw:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()
    df.index = df.index.tz_convert(ET)
    out = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )[["Open", "High", "Low", "Close", "Volume"]]
    out["Ticker"] = symbol.upper()
    return out.dropna(subset=["Close"])


def download_tiingo_iex_range(
    symbol: str = "SPY",
    *,
    start: date | None = None,
    end: date | None = None,
    after_hours: bool = True,
    chunk_days: int = 14,
) -> Path:
    """Chunked download into research parquet (does not touch Alpaca live cache)."""
    end = end or datetime.now(ET).date()
    start = start or (end - timedelta(days=200))
    parts: list[pd.DataFrame] = []
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=chunk_days - 1), end)
        print(f"[tiingo] {symbol} {cur} -> {nxt} afterHours={after_hours}", flush=True)
        try:
            chunk = fetch_tiingo_iex_5m(symbol, cur, nxt, after_hours=after_hours)
            if not chunk.empty:
                parts.append(chunk)
                print(f"  +{len(chunk)} bars", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  fail: {exc}", flush=True)
        cur = nxt + timedelta(days=1)
    if not parts:
        raise RuntimeError("No Tiingo bars downloaded")
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    path = tiingo_iex_path(symbol, after_hours=after_hours)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    hrs = sorted(pd.Series(out.index.hour).unique().tolist())
    print(f"[tiingo] wrote {path} n={len(out)} hours_ET={hrs} {out.index.min()} -> {out.index.max()}")
    return path


def load_tiingo_iex(symbol: str = "SPY", *, after_hours: bool = True) -> pd.DataFrame:
    path = tiingo_iex_path(symbol, after_hours=after_hours)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize(ET)
    return df


def filter_rth_et(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    t = df.index.time
    from datetime import time as dtime

    mask = [(x >= dtime(9, 30)) and (x < dtime(16, 0)) for x in t]
    return df.loc[mask].copy()
