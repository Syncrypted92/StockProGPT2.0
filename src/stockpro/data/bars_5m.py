"""Alpaca 5-minute bars with local parquet cache (SPY day lane)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from stockpro.config import ROOT, Settings, load_settings

RTH_START = (9, 30)
RTH_END = (16, 0)
# US equity extended: premarket 04:00–09:30 ET, after-hours 16:00–20:00 ET
PREMARKET_START = (4, 0)
PREMARKET_END = (9, 30)


def _bars_path(settings: Settings | None = None, symbol: str = "SPY") -> Path:
    settings = settings or load_settings()
    cfg = settings.get("spy_day", default={}) or {}
    sym = str(symbol).upper()
    if sym == "SPY":
        rel = cfg.get("bars_path") or f"data/bars/{sym}_5m.parquet"
    else:
        rel = f"data/bars/{sym}_5m.parquet"
    return ROOT / rel


def _ext_bars_path(settings: Settings | None = None, symbol: str = "SPY") -> Path:
    settings = settings or load_settings()
    cfg = settings.get("spy_day", default={}) or {}
    sym = str(symbol).upper()
    if sym == "SPY":
        rel = cfg.get("bars_ext_path") or f"data/bars/{sym}_5m_ext.parquet"
    else:
        rel = f"data/bars/{sym}_5m_ext.parquet"
    return ROOT / rel


def _normalize_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "Ticker"])
    out = df.copy()
    if isinstance(out.index, pd.MultiIndex):
        if "symbol" in out.index.names:
            out = out.reset_index(level=0, drop=True)
        else:
            out = out.droplevel(0)
    out.index = pd.to_datetime(out.index, utc=True).tz_convert("America/New_York")
    rename = {c: str(c).title() for c in out.columns}
    out = out.rename(columns=rename)
    colmap = {
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
        "Vwap": "VWAP",
        "Trade_Count": "Trade_Count",
    }
    for src, dst in list(colmap.items()):
        if src.lower() in {c.lower() for c in out.columns}:
            match = next(c for c in out.columns if c.lower() == src.lower())
            out = out.rename(columns={match: dst})
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume", "VWAP"] if c in out.columns]
    out = out[keep].sort_index()
    out["Ticker"] = symbol
    return out.dropna(subset=["Close"])


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep regular trading hours 09:30–16:00 America/New_York."""
    if df.empty:
        return df
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("America/New_York")
        df = df.copy()
        df.index = idx
    minutes = idx.hour * 60 + idx.minute
    start_m = RTH_START[0] * 60 + RTH_START[1]
    end_m = RTH_END[0] * 60 + RTH_END[1]
    mask = (minutes >= start_m) & (minutes < end_m) & (idx.dayofweek < 5)
    return df.loc[mask].copy()


def filter_premarket(df: pd.DataFrame) -> pd.DataFrame:
    """Keep premarket 04:00–09:30 ET (excludes the 09:30 RTH open bar)."""
    if df.empty:
        return df
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("America/New_York")
        df = df.copy()
        df.index = idx
    minutes = idx.hour * 60 + idx.minute
    start_m = PREMARKET_START[0] * 60 + PREMARKET_START[1]
    end_m = PREMARKET_END[0] * 60 + PREMARKET_END[1]
    mask = (minutes >= start_m) & (minutes < end_m) & (idx.dayofweek < 5)
    return df.loc[mask].copy()


def filter_extended_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep 04:00–20:00 ET weekdays (premarket + RTH + after-hours)."""
    if df.empty:
        return df
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("America/New_York")
        df = df.copy()
        df.index = idx
    minutes = idx.hour * 60 + idx.minute
    mask = (minutes >= 4 * 60) & (minutes < 20 * 60) & (idx.dayofweek < 5)
    return df.loc[mask].copy()


def download_alpaca_bars_5m(
    symbol: str = "SPY",
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    history_days: int = 180,
    api_key: str = "",
    secret_key: str = "",
    feed: str | None = None,
) -> pd.DataFrame:
    """Download 5-minute bars from Alpaca Market Data (includes extended hours when feed allows)."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed, Adjustment

    end = end or datetime.now(timezone.utc)
    start = start or (end - timedelta(days=int(history_days)))

    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    feeds_to_try: list[Any] = []
    if feed:
        feeds_to_try = [getattr(DataFeed, feed.upper(), DataFeed.IEX)]
    else:
        feeds_to_try = [DataFeed.IEX, DataFeed.SIP]

    last_exc: Exception | None = None
    bars = None
    used_feed = None
    for f in feeds_to_try:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=start,
                end=end,
                feed=f,
                adjustment=Adjustment.SPLIT,
            )
            bars = client.get_stock_bars(req)
            used_feed = f
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
    if bars is None:
        raise RuntimeError(f"Alpaca 5m bars failed for {symbol}: {last_exc}")

    df = bars.df if hasattr(bars, "df") else pd.DataFrame(bars)
    out = _normalize_bars(df, symbol)
    if used_feed is not None:
        print(f"[bars5m] feed={getattr(used_feed, 'value', used_feed)} bars={len(out)}")
    return out


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")
    return df.sort_index()


def load_cached_bars_5m(
    settings: Settings | None = None,
    symbol: str = "SPY",
) -> pd.DataFrame:
    return _load_parquet(_bars_path(settings, symbol))


def load_cached_bars_5m_ext(
    settings: Settings | None = None,
    symbol: str = "SPY",
) -> pd.DataFrame:
    return _load_parquet(_ext_bars_path(settings, symbol))


def save_bars_5m(df: pd.DataFrame, settings: Settings | None = None, symbol: str = "SPY") -> Path:
    path = _bars_path(settings, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if "Ticker" not in out.columns:
        out["Ticker"] = symbol
    out.to_parquet(path)
    return path


def save_bars_5m_ext(df: pd.DataFrame, settings: Settings | None = None, symbol: str = "SPY") -> Path:
    path = _ext_bars_path(settings, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if "Ticker" not in out.columns:
        out["Ticker"] = symbol
    out.to_parquet(path)
    return path


def refresh_bars_5m(
    settings: Settings | None = None,
    *,
    symbol: str | None = None,
    history_days: int | None = None,
    force_full: bool = False,
) -> pd.DataFrame:
    """
    Incremental (or full) refresh.
    Saves extended-hours cache + RTH-only cache used by the live ORB lane.
    Returns RTH bars (backward compatible).
    """
    settings = settings or load_settings()
    cfg = settings.get("spy_day", default={}) or {}
    symbol = symbol or str(cfg.get("symbol", "SPY"))
    history_days = int(history_days or cfg.get("history_days", 180))
    settings.require_broker_credentials()

    cached_ext = load_cached_bars_5m_ext(settings, symbol)
    if cached_ext.empty:
        cached_ext = load_cached_bars_5m(settings, symbol)  # migrate from old RTH-only cache

    end = datetime.now(timezone.utc)
    if force_full or cached_ext.empty:
        start = end - timedelta(days=history_days)
        print(f"[bars5m] full download {symbol} last {history_days}d (extended hours)...")
        fresh = download_alpaca_bars_5m(
            symbol,
            start=start,
            end=end,
            history_days=history_days,
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )
        combined = fresh
    else:
        start = (cached_ext.index.max().tz_convert("UTC") - timedelta(days=2)).to_pydatetime()
        print(f"[bars5m] incremental {symbol} from {start.date()} (extended hours)...")
        fresh = download_alpaca_bars_5m(
            symbol,
            start=start,
            end=end,
            history_days=history_days,
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )
        combined = pd.concat([cached_ext, fresh])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        cutoff = combined.index.max() - pd.Timedelta(days=history_days + 5)
        combined = combined.loc[combined.index >= cutoff]

    combined = filter_extended_session(combined)
    ext_path = save_bars_5m_ext(combined, settings, symbol)
    rth = filter_rth(combined)
    rth_path = save_bars_5m(rth, settings, symbol)

    pm = filter_premarket(combined)
    print(
        f"[bars5m] ext={len(combined)} ({combined.index.min()} -> {combined.index.max()}) -> {ext_path}"
    )
    print(f"[bars5m] premarket bars in cache: {len(pm)}")
    print(f"[bars5m] RTH={len(rth)} -> {rth_path}")
    return rth


def get_spy_5m(
    settings: Settings | None = None,
    *,
    refresh: bool = False,
    rth_only: bool = True,
) -> pd.DataFrame:
    """Load SPY 5m bars; optionally refresh from Alpaca first."""
    settings = settings or load_settings()
    cfg = settings.get("spy_day", default={}) or {}
    symbol = str(cfg.get("symbol", "SPY"))
    if refresh or not _bars_path(settings, symbol).exists():
        df = refresh_bars_5m(settings, symbol=symbol)
        if not rth_only:
            return load_cached_bars_5m_ext(settings, symbol)
        return df
    if rth_only:
        return filter_rth(load_cached_bars_5m(settings, symbol))
    ext = load_cached_bars_5m_ext(settings, symbol)
    if ext.empty:
        # Fall back to RTH cache if ext not downloaded yet
        return load_cached_bars_5m(settings, symbol)
    return ext


def get_spy_premarket_5m(
    settings: Settings | None = None,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load premarket-only 5m bars (04:00–09:30 ET)."""
    ext = get_spy_5m(settings, refresh=refresh, rth_only=False)
    return filter_premarket(ext)


def premarket_session_stats(ext_df: pd.DataFrame, session_date) -> dict[str, float] | None:
    """
    Premarket OHLC for a calendar date (ET).
    Returns None if no premarket bars that day.
    """
    if ext_df.empty:
        return None
    pm = filter_premarket(ext_df)
    day = pm[pm.index.date == session_date]
    if day.empty:
        return None
    return {
        "pm_open": float(day.iloc[0]["Open"]),
        "pm_high": float(day["High"].max()),
        "pm_low": float(day["Low"].min()),
        "pm_last": float(day.iloc[-1]["Close"]),
        "pm_volume": float(day["Volume"].sum()),
        "pm_bars": float(len(day)),
        "filled_in_pm": False,  # filled vs prior close set by caller
    }
