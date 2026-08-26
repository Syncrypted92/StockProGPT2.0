"""Technical indicator feature engineering (ported from notebook)."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "SMA_ratio",
    "SMA200_ratio",
    "EMA_ratio",
    "RSI",
    "MACD_hist",
    "BB_pct",
    "Stochastic",
    "ATR_pct",
    "realized_vol_20",
    "OBV_slope",
    "MFI",
    "CMF",
    "volume_z",
    # Regime / relative strength
    "SMA50_ratio",
    "trend_regime",
    "vol_regime",
    "rs_spy_5d",
    "rs_spy_20d",
]

# Logged daily by scan; merged at train time (same-day only — no lookahead)
NEWS_FEATURE_COLUMNS = [
    "news_score",
    "sentiment",
    "velocity",
    "headline_count",
    "bullish_hits",
    "bearish_hits",
    "weekend_bias",
]

ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + NEWS_FEATURE_COLUMNS


def _macd_histogram(close: pd.Series, short: int = 12, long: int = 26, signal: int = 9) -> pd.Series:
    ema_short = close.ewm(span=short, adjust=False).mean()
    ema_long = close.ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line


def _bollinger_pct(close: pd.Series, window: int = 20) -> pd.Series:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    width = (upper - lower).replace(0, np.nan)
    return (close - lower) / width


def _stochastic(df: pd.DataFrame, window: int = 14) -> pd.Series:
    low_min = df["Low"].rolling(window).min()
    high_max = df["High"].rolling(window).max()
    denom = (high_max - low_min).replace(0, np.nan)
    return 100 * (df["Close"] - low_min) / denom


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    money_flow = typical * df["Volume"]
    delta = typical.diff()
    pos = money_flow.where(delta > 0, 0.0)
    neg = money_flow.where(delta < 0, 0.0)
    pos_sum = pos.rolling(window).sum()
    neg_sum = neg.rolling(window).sum().replace(0, np.nan)
    ratio = pos_sum / neg_sum
    return 100 - (100 / (1 + ratio))


def _cmf(df: pd.DataFrame, window: int = 20) -> pd.Series:
    hl = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl
    mfv = mfm * df["Volume"]
    return mfv.rolling(window).sum() / df["Volume"].rolling(window).sum()


def add_indicators(
    df: pd.DataFrame,
    spy_close: pd.Series | None = None,
) -> pd.DataFrame:
    """Add technical indicators and model-ready feature columns."""
    out = df.copy()
    close = out["Close"]

    out["SMA"] = close.rolling(20).mean()
    out["EMA"] = close.ewm(span=12, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = gain / loss
    out["RSI"] = 100 - (100 / (1 + rs))

    out["MACD Histogram"] = _macd_histogram(close)
    out["Upper BB"] = out["SMA"] + 2 * close.rolling(20).std()
    out["Lower BB"] = out["SMA"] - 2 * close.rolling(20).std()
    out["Stochastic Oscillator"] = _stochastic(out)
    out["ATR"] = _atr(out)
    out["OBV"] = (np.sign(close.diff()).fillna(0) * out["Volume"]).cumsum()
    out["MFI"] = _mfi(out)
    out["CMF"] = _cmf(out)

    out["return_1d"] = close.pct_change(1)
    out["return_5d"] = close.pct_change(5)
    out["return_20d"] = close.pct_change(20)
    out["SMA_ratio"] = close / out["SMA"] - 1
    out["SMA200"] = close.rolling(200).mean()
    out["SMA200_ratio"] = close / out["SMA200"] - 1
    out["EMA_ratio"] = close / out["EMA"] - 1
    out["MACD_hist"] = out["MACD Histogram"] / close
    out["BB_pct"] = _bollinger_pct(close)
    out["Stochastic"] = out["Stochastic Oscillator"]
    out["ATR_pct"] = out["ATR"] / close
    out["realized_vol_20"] = close.pct_change().rolling(20).std()
    out["OBV_slope"] = out["OBV"].pct_change(5).replace([np.inf, -np.inf], np.nan)
    vol_mean = out["Volume"].rolling(20).mean()
    vol_std = out["Volume"].rolling(20).std().replace(0, np.nan)
    out["volume_z"] = (out["Volume"] - vol_mean) / vol_std

    # Regime features
    out["SMA50"] = close.rolling(50).mean()
    out["SMA50_ratio"] = close / out["SMA50"] - 1
    out["trend_regime"] = np.sign(out["SMA50"] - out["SMA200"]).fillna(0.0)
    vol_med = out["realized_vol_20"].rolling(60).median().replace(0, np.nan)
    out["vol_regime"] = out["realized_vol_20"] / vol_med

    if spy_close is not None and len(spy_close):
        spy = spy_close.reindex(out.index)
        out["rs_spy_5d"] = out["return_5d"] - spy.pct_change(5)
        out["rs_spy_20d"] = out["return_20d"] - spy.pct_change(20)
    else:
        out["rs_spy_5d"] = 0.0
        out["rs_spy_20d"] = 0.0

    return out


def add_forward_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    flat_threshold: float = 0.005,
    *,
    binary: bool = False,
) -> pd.DataFrame:
    """Add forward return and directional label: 1=up, -1=down, 0=flat (or NaN if binary)."""
    out = df.copy()
    out["forward_return"] = out["Close"].pct_change(horizon).shift(-horizon)
    if binary:
        label = np.where(
            out["forward_return"] > flat_threshold,
            1,
            np.where(out["forward_return"] < -flat_threshold, -1, np.nan),
        )
    else:
        label = np.where(
            out["forward_return"] > flat_threshold,
            1,
            np.where(out["forward_return"] < -flat_threshold, -1, 0),
        )
    out["direction"] = label
    return out


def feature_matrix(df: pd.DataFrame, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Return X, y for rows with complete features and labels."""
    cols = feature_cols or FEATURE_COLUMNS
    needed = cols + ["direction"]
    clean = df.dropna(subset=[c for c in needed if c in df.columns]).copy()
    present = [c for c in cols if c in clean.columns]
    return clean[present], clean["direction"]
