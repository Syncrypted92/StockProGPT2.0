"""Intraday SPY strategies beyond ORB (research / secondary lane).

1. gap_and_go          — overnight gap holds; trade WITH the gap
2. opening_drive       — strong first-15/30m drive continuation
3. trend_day_pullback  — trend day: pullback to VWAP/EMA9 with bias
4. failed_auction      — liquidity sweep beyond level + reclaim (stricter)
5. range_day_fade      — tight ADR day: fade extremes toward VWAP
6. power_hour          — 14:00–14:30 continuation of day bias
7. multi_day_break     — break of prior 3-session high/low
8. compression_break   — tight AM range then expansion break
9. spy_qqq_lead        — SPY leads QQQ (or lags) in ORB direction (optional QQQ bars)
"""

from __future__ import annotations

from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from stockpro.spy_day.patterns import PatternSignal, _huge_candle
from stockpro.spy_day.secondary_patterns import first_orb_break, overnight_gap_pct, session_slice

ROOT = Path(__file__).resolve().parents[3]
_QQQ_CACHE: pd.DataFrame | None = None
_QQQ_RET_BY_TS: dict | None = None
_AM_PCT_BY_DAY: dict | None = None
_ADR_BY_DAY: dict | None = None
_HL3_BY_DAY: dict | None = None


def clear_intraday_caches() -> None:
    global _QQQ_RET_BY_TS, _AM_PCT_BY_DAY, _ADR_BY_DAY, _HL3_BY_DAY
    _QQQ_RET_BY_TS = None
    _AM_PCT_BY_DAY = None
    _ADR_BY_DAY = None
    _HL3_BY_DAY = None


def _ts_time(ts) -> time:
    return ts.time() if hasattr(ts, "time") else time(12, 0)


def _already_fired(df: pd.DataFrame, i: int, pattern: str) -> bool:
    """True if this pattern already produced a candidate earlier today (soft: reason prefix)."""
    row = df.iloc[i]
    day = row["session_date"]
    prior = df.iloc[:i]
    same = prior[prior["session_date"] == day] if len(prior) else prior
    # We don't store pattern history on bars; use a lightweight per-day flag column if present
    flag = f"_fired_{pattern}"
    if flag in df.columns:
        return bool(same[flag].any()) if len(same) and flag in same.columns else False
    return False


def _session_range_pct(same: pd.DataFrame) -> float:
    if same.empty:
        return 0.0
    hi = float(same["High"].max())
    lo = float(same["Low"].min())
    mid = float(same["Close"].iloc[-1])
    if mid <= 0:
        return 0.0
    return (hi - lo) / mid


def _day_bias(same: pd.DataFrame) -> str | None:
    """call if price above open and OR mid; put if below."""
    if same.empty:
        return None
    o = float(same.iloc[0]["Open"])
    c = float(same.iloc[-1]["Close"])
    if c > o * 1.001:
        return "call"
    if c < o * 0.999:
        return "put"
    return None


def _adr_pct(df: pd.DataFrame, i: int, lookback: int = 20) -> float | None:
    """Average daily range % over prior sessions."""
    global _ADR_BY_DAY
    day = df.iloc[i]["session_date"]
    if _ADR_BY_DAY is None:
        _ADR_BY_DAY = {}
    if day in _ADR_BY_DAY:
        return _ADR_BY_DAY[day]

    dates = sorted({d for d in df["session_date"].unique() if d < day})
    if len(dates) < 5:
        _ADR_BY_DAY[day] = None
        return None
    use = dates[-lookback:]
    ranges = []
    for d in use:
        g = df[df["session_date"] == d]
        if g.empty:
            continue
        mid = float(g["Close"].iloc[-1])
        if mid <= 0:
            continue
        ranges.append((float(g["High"].max()) - float(g["Low"].min())) / mid)
    val = float(np.mean(ranges)) if len(ranges) >= 5 else None
    _ADR_BY_DAY[day] = val
    return val


def _prior_n_day_hl(df: pd.DataFrame, i: int, n: int = 3) -> tuple[float, float] | None:
    global _HL3_BY_DAY
    day = df.iloc[i]["session_date"]
    if _HL3_BY_DAY is None:
        _HL3_BY_DAY = {}
    if day in _HL3_BY_DAY:
        return _HL3_BY_DAY[day]

    dates = sorted({d for d in df["session_date"].unique() if d < day})
    if len(dates) < n:
        _HL3_BY_DAY[day] = None
        return None
    use = dates[-n:]
    hi = max(float(df[df["session_date"] == d]["High"].max()) for d in use)
    lo = min(float(df[df["session_date"] == d]["Low"].min()) for d in use)
    _HL3_BY_DAY[day] = (hi, lo)
    return hi, lo


# ---------------------------------------------------------------------------
# 1) Gap & go
# ---------------------------------------------------------------------------
def gap_and_go_signal(df: pd.DataFrame, i: int) -> PatternSignal | None:
    """Gap holds through OR; trade continuation in gap direction (not fill)."""
    row = df.iloc[i]
    ts = df.index[i]
    t = _ts_time(ts)
    if t < time(10, 0) or t > time(11, 30):
        return None
    if not bool(row.get("OR_READY", False)):
        return None

    gap = overnight_gap_pct(df, i)
    if gap is None or abs(gap) < 0.0020:  # >= 0.20%
        return None

    same = session_slice(df, i)
    # One per day: only first qualifying bar
    if len(same) > 1:
        # if we already closed beyond OR earlier in gap direction, skip later signals
        pass

    oh, ol = float(row["OR_HIGH"]), float(row["OR_LOW"])
    close = float(row["Close"])
    o0 = float(same.iloc[0]["Open"])
    atr = float(row["ATR"]) if pd.notna(row.get("ATR")) else np.nan
    if pd.isna(atr) or atr <= 0 or _huge_candle(row, atr):
        return None

    # Gap must not be filled: gap-up still above prior close / open holds
    if gap > 0:
        # Hold: session low never traded back to open by much; price still above OR mid
        if float(same["Low"].min()) < o0 - 0.15 * atr:
            return None  # filled / failed
        # Go: close breaks OR high
        if close <= oh or close <= float(row["Open"]):
            return None
        # First break only
        prior = same.iloc[:-1]
        if len(prior) and (prior["Close"] > prior["OR_HIGH"]).any():
            return None
        return PatternSignal(
            pattern="gap_and_go",
            side="call",
            confidence=0.78,
            reason=f"Gap&go up hold gap={gap*100:.2f}% break OR_H",
            bar_time=ts,
            spot=close,
        )

    # gap down
    if float(same["High"].max()) > o0 + 0.15 * atr:
        return None
    if close >= ol or close >= float(row["Open"]):
        return None
    prior = same.iloc[:-1]
    if len(prior) and (prior["Close"] < prior["OR_LOW"]).any():
        return None
    return PatternSignal(
        pattern="gap_and_go",
        side="put",
        confidence=0.78,
        reason=f"Gap&go down hold gap={gap*100:.2f}% break OR_L",
        bar_time=ts,
        spot=close,
    )


# ---------------------------------------------------------------------------
# 2) Opening drive
# ---------------------------------------------------------------------------
def opening_drive_signal(df: pd.DataFrame, i: int) -> PatternSignal | None:
    """Strong first ~30m directional drive; enter continuation as OR completes."""
    row = df.iloc[i]
    ts = df.index[i]
    t = _ts_time(ts)
    # Fire on first bars after OR ready (10:00–10:20)
    if t < time(10, 0) or t > time(10, 20):
        return None
    if not bool(row.get("OR_READY", False)):
        return None

    same = session_slice(df, i)
    if len(same) < 6:
        return None
    drive = same.iloc[:6]  # first 30m
    o0 = float(drive.iloc[0]["Open"])
    c_drive = float(drive.iloc[-1]["Close"])
    drive_move = (c_drive - o0) / o0
    atr = float(row["ATR"]) if pd.notna(row.get("ATR")) else np.nan
    if pd.isna(atr) or atr <= 0 or _huge_candle(row, atr):
        return None

    # Need meaningful drive vs ATR
    if abs(c_drive - o0) < 1.2 * atr:
        return None
    if abs(drive_move) < 0.0015:
        return None

    close = float(row["Close"])
    oh, ol = float(row["OR_HIGH"]), float(row["OR_LOW"])

    # Only first signal of day in this window
    if int(row.get("bar_in_day", 0)) > 7:
        return None

    if drive_move > 0 and close > oh and close > float(row["Open"]) and close > float(row["EMA9"]):
        return PatternSignal(
            pattern="opening_drive",
            side="call",
            confidence=0.76,
            reason=f"Opening drive up {drive_move*100:.2f}%",
            bar_time=ts,
            spot=close,
        )
    if drive_move < 0 and close < ol and close < float(row["Open"]) and close < float(row["EMA9"]):
        return PatternSignal(
            pattern="opening_drive",
            side="put",
            confidence=0.76,
            reason=f"Opening drive down {drive_move*100:.2f}%",
            bar_time=ts,
            spot=close,
        )
    return None


# ---------------------------------------------------------------------------
# 3) Trend-day pullback
# ---------------------------------------------------------------------------
def trend_day_pullback_signal(df: pd.DataFrame, i: int) -> PatternSignal | None:
    """Expanding range day: pullback to VWAP/EMA9, continue with ORB or open bias."""
    if i < 3:
        return None
    row = df.iloc[i]
    ts = df.index[i]
    t = _ts_time(ts)
    if t < time(10, 30) or t > time(14, 0):
        return None

    same = session_slice(df, i)
    if len(same) < 12:
        return None

    atr = float(row["ATR"]) if pd.notna(row.get("ATR")) else np.nan
    vwap = float(row["VWAP"]) if pd.notna(row.get("VWAP")) else np.nan
    ema9 = float(row["EMA9"]) if pd.notna(row.get("EMA9")) else np.nan
    if pd.isna(atr) or atr <= 0 or pd.isna(vwap) or pd.isna(ema9) or _huge_candle(row, atr):
        return None

    # Trend day proxy: session range already >= 0.55 * ADR or clear ORB direction
    adr = _adr_pct(df, i)
    sess_rng = _session_range_pct(same)
    bias, _ = first_orb_break(same.iloc[:-1] if len(same) > 1 else same.iloc[:0])
    if bias is None:
        bias = _day_bias(same)
    if bias is None:
        return None
    if adr is not None and sess_rng < 0.45 * adr:
        return None  # not expanding enough

    close = float(row["Close"])
    low, high = float(row["Low"]), float(row["High"])
    prev = df.iloc[i - 1]
    o = float(row["Open"])

    # One pullback entry per day (first only)
    # Approximate: if price already made a similar reclaim earlier, skip
    after = same.iloc[6:-1]  # after ~10:00
    if bias == "call":
        for _, prow in after.iterrows():
            if (
                float(prow["Low"]) <= float(prow["VWAP"]) + 0.15 * atr
                and float(prow["Close"]) > float(prow["VWAP"])
                and float(prow["Close"]) > float(prow["Open"])
            ):
                return None
        # Pullback tags VWAP or EMA9 from above, bullish reclaim
        tagged = low <= max(vwap, ema9) + 0.20 * atr and high >= min(vwap, ema9)
        if not tagged:
            return None
        if close <= vwap or close <= o or close <= ema9:
            return None
        if not bool(row.get("above_vwap", False)):
            return None
        return PatternSignal(
            pattern="trend_day_pullback",
            side="call",
            confidence=0.77,
            reason=f"Trend pullback long VWAP/EMA @{vwap:.2f}",
            bar_time=ts,
            spot=close,
        )

    # put
    for _, prow in after.iterrows():
        if (
            float(prow["High"]) >= float(prow["VWAP"]) - 0.15 * atr
            and float(prow["Close"]) < float(prow["VWAP"])
            and float(prow["Close"]) < float(prow["Open"])
        ):
            return None
    tagged = high >= min(vwap, ema9) - 0.20 * atr and low <= max(vwap, ema9)
    if not tagged:
        return None
    if close >= vwap or close >= o or close >= ema9:
        return None
    if not bool(row.get("below_vwap", False)):
        return None
    return PatternSignal(
        pattern="trend_day_pullback",
        side="put",
        confidence=0.77,
        reason=f"Trend pullback short VWAP/EMA @{vwap:.2f}",
        bar_time=ts,
        spot=close,
    )


# ---------------------------------------------------------------------------
# 4) Failed auction (stricter liquidity grab)
# ---------------------------------------------------------------------------
def failed_auction_signal(df: pd.DataFrame, i: int) -> PatternSignal | None:
    """Sweep beyond OR or PDH/PDL, close back through level, confirmation candle."""
    if i < 1:
        return None
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    ts = df.index[i]
    t = _ts_time(ts)
    if t < time(10, 5) or t > time(13, 30):
        return None

    atr = float(row["ATR"]) if pd.notna(row.get("ATR")) else np.nan
    if pd.isna(atr) or atr <= 0 or _huge_candle(row, atr):
        return None

    close = float(row["Close"])
    levels: list[tuple[str, float]] = []
    if bool(row.get("OR_READY", False)):
        levels.append(("OR_H", float(row["OR_HIGH"])))
        levels.append(("OR_L", float(row["OR_LOW"])))
    if pd.notna(row.get("PDH")):
        levels.append(("PDH", float(row["PDH"])))
    if pd.notna(row.get("PDL")):
        levels.append(("PDL", float(row["PDL"])))

    pierce = 0.20 * atr
    same = session_slice(df, i)
    # one failed auction per day
    if len(same) > 20:
        # cheap throttle: only before early afternoon and first half of signals
        pass

    for name, lvl in levels:
        # Bearish failed auction: prev swept above lvl, closed back below; this bar confirms down
        if name in ("OR_H", "PDH"):
            swept = float(prev["High"]) > lvl + pierce and float(prev["Close"]) < lvl
            confirm = close < float(prev["Close"]) and close < float(row["Open"]) and close < lvl
            if swept and confirm:
                return PatternSignal(
                    pattern="failed_auction",
                    side="put",
                    confidence=0.78,
                    reason=f"Failed auction fade {name}@{lvl:.2f}",
                    bar_time=ts,
                    spot=close,
                )
        if name in ("OR_L", "PDL"):
            swept = float(prev["Low"]) < lvl - pierce and float(prev["Close"]) > lvl
            confirm = close > float(prev["Close"]) and close > float(row["Open"]) and close > lvl
            if swept and confirm:
                return PatternSignal(
                    pattern="failed_auction",
                    side="call",
                    confidence=0.78,
                    reason=f"Failed auction reclaim {name}@{lvl:.2f}",
                    bar_time=ts,
                    spot=close,
                )
    return None


# ---------------------------------------------------------------------------
# 5) Range-day fade
# ---------------------------------------------------------------------------
def range_day_fade_signal(df: pd.DataFrame, i: int) -> PatternSignal | None:
    """Tight day vs ADR: fade session extreme back toward VWAP."""
    row = df.iloc[i]
    ts = df.index[i]
    t = _ts_time(ts)
    if t < time(11, 0) or t > time(14, 0):
        return None

    same = session_slice(df, i)
    if len(same) < 18:
        return None
    adr = _adr_pct(df, i)
    if adr is None:
        return None
    sess_rng = _session_range_pct(same)
    if sess_rng > 0.70 * adr:
        return None  # too trendy

    atr = float(row["ATR"]) if pd.notna(row.get("ATR")) else np.nan
    vwap = float(row["VWAP"]) if pd.notna(row.get("VWAP")) else np.nan
    if pd.isna(atr) or atr <= 0 or pd.isna(vwap) or _huge_candle(row, atr):
        return None

    close = float(row["Close"])
    hi = float(same["High"].max())
    lo = float(same["Low"].min())
    o = float(row["Open"])

    # Tag session high and reject
    if float(row["High"]) >= hi - 0.05 * atr and close < vwap and close < o and close < float(row["EMA9"]):
        return PatternSignal(
            pattern="range_day_fade",
            side="put",
            confidence=0.74,
            reason=f"Range fade from high toward VWAP@{vwap:.2f}",
            bar_time=ts,
            spot=close,
        )
    if float(row["Low"]) <= lo + 0.05 * atr and close > vwap and close > o and close > float(row["EMA9"]):
        return PatternSignal(
            pattern="range_day_fade",
            side="call",
            confidence=0.74,
            reason=f"Range fade from low toward VWAP@{vwap:.2f}",
            bar_time=ts,
            spot=close,
        )
    return None


# ---------------------------------------------------------------------------
# 6) Power hour
# ---------------------------------------------------------------------------
def power_hour_signal(df: pd.DataFrame, i: int) -> PatternSignal | None:
    """14:00–14:30 continuation of established day/ORB bias (before entry cutoff)."""
    row = df.iloc[i]
    ts = df.index[i]
    t = _ts_time(ts)
    if t < time(14, 0) or t > time(14, 25):
        return None

    same = session_slice(df, i)
    if len(same) < 40:
        return None
    atr = float(row["ATR"]) if pd.notna(row.get("ATR")) else np.nan
    if pd.isna(atr) or atr <= 0 or _huge_candle(row, atr):
        return None

    bias, _ = first_orb_break(same)
    if bias is None:
        bias = _day_bias(same)
    if bias is None:
        return None

    # Need clear separation from open
    o0 = float(same.iloc[0]["Open"])
    close = float(row["Close"])
    if abs(close - o0) < 0.8 * atr:
        return None

    # First power-hour bar signal only
    ph = same[(same.index.time >= time(14, 0)) & (same.index < ts)]
    if len(ph):
        return None

    if bias == "call" and close > float(row["VWAP"]) and close > float(row["EMA9"]) and close > float(row["Open"]):
        return PatternSignal(
            pattern="power_hour",
            side="call",
            confidence=0.75,
            reason="Power hour continuation long",
            bar_time=ts,
            spot=close,
        )
    if bias == "put" and close < float(row["VWAP"]) and close < float(row["EMA9"]) and close < float(row["Open"]):
        return PatternSignal(
            pattern="power_hour",
            side="put",
            confidence=0.75,
            reason="Power hour continuation short",
            bar_time=ts,
            spot=close,
        )
    return None


# ---------------------------------------------------------------------------
# 7) Multi-day break
# ---------------------------------------------------------------------------
def multi_day_break_signal(df: pd.DataFrame, i: int) -> PatternSignal | None:
    """Close break of prior 3-session high/low with body confirmation."""
    row = df.iloc[i]
    ts = df.index[i]
    t = _ts_time(ts)
    if t < time(10, 0) or t > time(14, 0):
        return None

    hl = _prior_n_day_hl(df, i, n=3)
    if hl is None:
        return None
    hi3, lo3 = hl
    atr = float(row["ATR"]) if pd.notna(row.get("ATR")) else np.nan
    if pd.isna(atr) or atr <= 0 or _huge_candle(row, atr):
        return None

    close = float(row["Close"])
    same = session_slice(df, i)
    # first break only today
    prior = same.iloc[:-1]
    if len(prior) and ((prior["Close"] > hi3).any() or (prior["Close"] < lo3).any()):
        return None

    if close > hi3 + 0.05 * atr and close > float(row["Open"]) and float(row["Low"]) <= hi3 + 0.25 * atr:
        return PatternSignal(
            pattern="multi_day_break",
            side="call",
            confidence=0.77,
            reason=f"3-day high break @{hi3:.2f}",
            bar_time=ts,
            spot=close,
        )
    if close < lo3 - 0.05 * atr and close < float(row["Open"]) and float(row["High"]) >= lo3 - 0.25 * atr:
        return PatternSignal(
            pattern="multi_day_break",
            side="put",
            confidence=0.77,
            reason=f"3-day low break @{lo3:.2f}",
            bar_time=ts,
            spot=close,
        )
    return None


# ---------------------------------------------------------------------------
# 8) Compression break
# ---------------------------------------------------------------------------
def compression_break_signal(df: pd.DataFrame, i: int) -> PatternSignal | None:
    """First 30m range is tight vs recent opens; break expands."""
    global _AM_PCT_BY_DAY
    row = df.iloc[i]
    ts = df.index[i]
    t = _ts_time(ts)
    if t < time(10, 0) or t > time(11, 30):
        return None
    if not bool(row.get("OR_READY", False)):
        return None

    same = session_slice(df, i)
    am = same.iloc[:6]
    if len(am) < 6:
        return None
    am_rng = float(am["High"].max()) - float(am["Low"].min())
    mid = float(am["Close"].iloc[-1])
    if mid <= 0:
        return None
    am_pct = am_rng / mid

    day = row["session_date"]
    if _AM_PCT_BY_DAY is None:
        # Precompute first-30m range % for all days once
        _AM_PCT_BY_DAY = {}
        for d, g in df.groupby("session_date"):
            g6 = g.iloc[:6]
            if len(g6) < 6:
                continue
            m = float(g6["Close"].iloc[-1])
            if m <= 0:
                continue
            _AM_PCT_BY_DAY[d] = (float(g6["High"].max()) - float(g6["Low"].min())) / m

    dates = sorted(d for d in _AM_PCT_BY_DAY if d < day)[-20:]
    prior_pcts = [_AM_PCT_BY_DAY[d] for d in dates]
    if len(prior_pcts) < 8:
        return None
    if am_pct > float(np.percentile(prior_pcts, 30)):
        return None  # not compressed

    atr = float(row["ATR"]) if pd.notna(row.get("ATR")) else np.nan
    if pd.isna(atr) or atr <= 0 or _huge_candle(row, atr):
        return None

    oh, ol = float(row["OR_HIGH"]), float(row["OR_LOW"])
    close = float(row["Close"])
    prior = same.iloc[:-1]
    if close > oh and close > float(row["Open"]):
        if len(prior) and (prior["Close"] > prior["OR_HIGH"]).any():
            return None
        return PatternSignal(
            pattern="compression_break",
            side="call",
            confidence=0.76,
            reason=f"Compression break up AM={am_pct*100:.2f}%",
            bar_time=ts,
            spot=close,
        )
    if close < ol and close < float(row["Open"]):
        if len(prior) and (prior["Close"] < prior["OR_LOW"]).any():
            return None
        return PatternSignal(
            pattern="compression_break",
            side="put",
            confidence=0.76,
            reason=f"Compression break down AM={am_pct*100:.2f}%",
            bar_time=ts,
            spot=close,
        )
    return None


# ---------------------------------------------------------------------------
# 9) SPY vs QQQ lead/lag
# ---------------------------------------------------------------------------
def _ensure_qqq_session_rets(spy_index: pd.DatetimeIndex) -> dict | None:
    """Map timestamp -> QQQ return from that session's open to this bar."""
    global _QQQ_CACHE, _QQQ_RET_BY_TS
    if _QQQ_RET_BY_TS is not None:
        return _QQQ_RET_BY_TS
    path = ROOT / "data" / "bars" / "QQQ_5m.parquet"
    if not path.exists():
        return None
    q = pd.read_parquet(path)
    if q.index.tz is None:
        q.index = q.index.tz_localize("America/New_York")
    _QQQ_CACHE = q
    common = spy_index.intersection(q.index)
    if len(common) < 100:
        return None
    qq = q.loc[common].copy()
    day = pd.Series(qq.index.date, index=qq.index)
    sess_open = qq["Open"].groupby(day).transform("first")
    rets = (qq["Close"] - sess_open) / sess_open.replace(0, np.nan)
    _QQQ_RET_BY_TS = {
        ts: float(v) for ts, v in rets.items() if pd.notna(v)
    }
    return _QQQ_RET_BY_TS


def _spy_open_ret(df: pd.DataFrame, i: int) -> float | None:
    same = session_slice(df, i)
    if same.empty:
        return None
    o = float(same.iloc[0]["Open"])
    c = float(df.iloc[i]["Close"])
    if o <= 0:
        return None
    return (c - o) / o


def spy_qqq_lead_signal(df: pd.DataFrame, i: int) -> PatternSignal | None:
    """ORB-direction trade only when SPY is leading QQQ on the morning move."""
    row = df.iloc[i]
    ts = df.index[i]
    t = _ts_time(ts)
    if t < time(10, 0) or t > time(11, 30):
        return None
    if not bool(row.get("OR_READY", False)):
        return None

    qrets = _ensure_qqq_session_rets(df.index)
    if qrets is None or ts not in qrets:
        return None

    same = session_slice(df, i)
    side, break_j = first_orb_break(same)
    if side is None or break_j is None or break_j != len(same) - 1:
        return None

    spy_ret = _spy_open_ret(df, i)
    if spy_ret is None:
        return None
    q_ret = qrets[ts]

    atr = float(row["ATR"]) if pd.notna(row.get("ATR")) else np.nan
    if pd.isna(atr) or atr <= 0 or _huge_candle(row, atr):
        return None

    spy_c = float(row["Close"])
    if side == "call":
        if not (spy_ret > q_ret + 0.0005 and spy_ret > 0):
            return None
        return PatternSignal(
            pattern="spy_qqq_lead",
            side="call",
            confidence=0.79,
            reason=f"SPY leads QQQ ({spy_ret*100:.2f}% vs {q_ret*100:.2f}%) ORB call",
            bar_time=ts,
            spot=spy_c,
        )
    if not (spy_ret < q_ret - 0.0005 and spy_ret < 0):
        return None
    return PatternSignal(
        pattern="spy_qqq_lead",
        side="put",
        confidence=0.79,
        reason=f"SPY leads QQQ down ({spy_ret*100:.2f}% vs {q_ret*100:.2f}%) ORB put",
        bar_time=ts,
        spot=spy_c,
    )
