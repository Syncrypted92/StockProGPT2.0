"""Secondary SPY day patterns (separate from ORB core / OTE).

1. orb_gap_align   — ORB only in overnight gap-fill direction
2. orb_retest      — pullback to OR after first break, continuation
3. vwap_orb_bias   — VWAP reclaim/reject only after ORB direction set
4. pdh_pdl_killzone — PDH/PDL reaction in NY morning killzone
5. ib_failure      — false OR break that closes back inside (fade)
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from stockpro.spy_day.patterns import PatternSignal, _huge_candle


def session_slice(df: pd.DataFrame, i: int) -> pd.DataFrame:
    day = df.iloc[i]["session_date"]
    return df.iloc[: i + 1][df.iloc[: i + 1]["session_date"] == day]


def first_orb_break(same: pd.DataFrame) -> tuple[str | None, int | None]:
    """Return ('call'|'put', local_iloc) of first OR close break in this session frame."""
    if same.empty or "OR_READY" not in same.columns:
        return None, None
    for j in range(len(same)):
        row = same.iloc[j]
        if not bool(row.get("OR_READY", False)):
            continue
        oh, ol = float(row["OR_HIGH"]), float(row["OR_LOW"])
        c = float(row["Close"])
        if c > oh:
            return "call", j
        if c < ol:
            return "put", j
    return None, None


def overnight_gap_pct(df: pd.DataFrame, i: int) -> float | None:
    """(today open - prior RTH close) / prior close."""
    row = df.iloc[i]
    day = row["session_date"]
    # Prefer precomputed column
    if "GAP_PCT" in df.columns and pd.notna(row.get("GAP_PCT")):
        return float(row["GAP_PCT"])
    # Fallback
    dates = sorted({d for d in df["session_date"].unique()})
    if day not in dates:
        return None
    idx = dates.index(day)
    if idx == 0:
        return None
    prev_day = dates[idx - 1]
    prev = df[df["session_date"] == prev_day]
    today = df[df["session_date"] == day]
    if prev.empty or today.empty:
        return None
    pc = float(prev.iloc[-1]["Close"])
    o = float(today.iloc[0]["Open"])
    if pc <= 0:
        return None
    return (o - pc) / pc


# ---------------------------------------------------------------------------
# 1) Gap + ORB align (filter flavor: only ORB breaks that fade the gap)
# ---------------------------------------------------------------------------
def orb_gap_align_signal(df: pd.DataFrame, i: int) -> PatternSignal | None:
    """ORB break only if it is in overnight gap-fill direction (gap-up→put, gap-down→call)."""
    from stockpro.spy_day.patterns import _orb_signal

    gap = overnight_gap_pct(df, i)
    if gap is None or abs(gap) < 0.0015:
        return None
    sig = _orb_signal(df, i)
    if sig is None or sig.pattern != "orb":
        return None
    # Gap up → want put (fill down); gap down → want call
    want = "put" if gap > 0 else "call"
    if sig.side != want:
        return None
    return PatternSignal(
        pattern="orb_gap_align",
        side=sig.side,
        confidence=min(sig.confidence + 0.03, 0.95),
        reason=f"{sig.reason} gap={gap*100:+.2f}% fill-align",
        bar_time=sig.bar_time,
        spot=sig.spot,
    )


# ---------------------------------------------------------------------------
# 2) ORB retest — continuation after pullback to OR level
# ---------------------------------------------------------------------------
def orb_retest_signal(df: pd.DataFrame, i: int) -> PatternSignal | None:
    row = df.iloc[i]
    ts = df.index[i]
    t = ts.time() if hasattr(ts, "time") else time(12, 0)
    if t < time(10, 15) or t > time(13, 30):
        return None
    if not bool(row.get("OR_READY", False)):
        return None

    same = session_slice(df, i)
    side, break_j = first_orb_break(same.iloc[:-1] if len(same) > 1 else same.iloc[:0])
    if side is None or break_j is None:
        return None
    # Need at least 1 bar after break before retest
    if len(same) - 1 <= break_j + 1:
        return None

    oh, ol = float(row["OR_HIGH"]), float(row["OR_LOW"])
    close = float(row["Close"])
    high, low = float(row["High"]), float(row["Low"])
    atr = float(row["ATR"]) if pd.notna(row.get("ATR")) else np.nan
    if pd.isna(atr) or atr <= 0 or _huge_candle(row, atr):
        return None

    # One retest trade per day
    day = row["session_date"]
    prior = df.iloc[:i]
    # Soft: if price already made a confirmed retest earlier — skip via close beyond OR again? keep simple: first retest only
    # Detect prior retest: after break, a bar that tagged OR and closed in direction
    after_break = same.iloc[break_j + 1 : -1]
    for _, prow in after_break.iterrows():
        if side == "call" and float(prow["Low"]) <= oh + 0.15 * atr and float(prow["Close"]) > oh:
            return None
        if side == "put" and float(prow["High"]) >= ol - 0.15 * atr and float(prow["Close"]) < ol:
            return None

    if side == "call":
        # Pullback tags OR high from above, bullish close back above
        tagged = low <= oh + 0.20 * atr and high >= oh
        if not tagged or close <= oh or close <= float(row["Open"]):
            return None
        # Still above OR (structure intact)
        if float(same.iloc[break_j + 1 :]["Close"].min()) < ol:
            return None
        return PatternSignal(
            pattern="orb_retest",
            side="call",
            confidence=0.76,
            reason=f"ORB retest hold @{oh:.2f}",
            bar_time=ts,
            spot=close,
        )

    # put
    tagged = high >= ol - 0.20 * atr and low <= ol
    if not tagged or close >= ol or close >= float(row["Open"]):
        return None
    if float(same.iloc[break_j + 1 :]["Close"].max()) > oh:
        return None
    return PatternSignal(
        pattern="orb_retest",
        side="put",
        confidence=0.76,
        reason=f"ORB retest hold @{ol:.2f}",
        bar_time=ts,
        spot=close,
    )


# ---------------------------------------------------------------------------
# 3) VWAP reclaim with ORB bias
# ---------------------------------------------------------------------------
def vwap_orb_bias_signal(df: pd.DataFrame, i: int) -> PatternSignal | None:
    if i < 3:
        return None
    row = df.iloc[i]
    ts = df.index[i]
    t = ts.time() if hasattr(ts, "time") else time(12, 0)
    if t < time(10, 15) or t > time(14, 0):
        return None

    same = session_slice(df, i)
    bias, break_j = first_orb_break(same.iloc[:-1] if len(same) > 1 else same.iloc[:0])
    if bias is None:
        return None

    prev, prev2 = df.iloc[i - 1], df.iloc[i - 2]
    close = float(row["Close"])
    vwap = float(row["VWAP"]) if pd.notna(row.get("VWAP")) else np.nan
    atr = float(row["ATR"]) if pd.notna(row.get("ATR")) else np.nan
    if pd.isna(vwap) or pd.isna(atr) or atr <= 0 or _huge_candle(row, atr):
        return None

    if bias == "call":
        if not (
            bool(prev2.get("below_vwap", False))
            and bool(prev.get("below_vwap", False))
            and close > vwap
            and close > float(row["Open"])
        ):
            return None
        return PatternSignal(
            pattern="vwap_orb_bias",
            side="call",
            confidence=0.74,
            reason=f"VWAP reclaim w/ ORB-call bias @{vwap:.2f}",
            bar_time=ts,
            spot=close,
        )

    if not (
        bool(prev2.get("above_vwap", False))
        and bool(prev.get("above_vwap", False))
        and close < vwap
        and close < float(row["Open"])
    ):
        return None
    return PatternSignal(
        pattern="vwap_orb_bias",
        side="put",
        confidence=0.74,
        reason=f"VWAP reject w/ ORB-put bias @{vwap:.2f}",
        bar_time=ts,
        spot=close,
    )


# ---------------------------------------------------------------------------
# 4) PDH/PDL killzone reaction
# ---------------------------------------------------------------------------
def pdh_pdl_killzone_signal(df: pd.DataFrame, i: int) -> PatternSignal | None:
    row = df.iloc[i]
    ts = df.index[i]
    t = ts.time() if hasattr(ts, "time") else time(12, 0)
    # NY morning killzone
    if not (time(10, 0) <= t <= time(11, 30)):
        return None

    pdh, pdl = row.get("PDH"), row.get("PDL")
    if pd.isna(pdh) or pd.isna(pdl):
        return None
    pdh, pdl = float(pdh), float(pdl)
    close = float(row["Close"])
    high, low = float(row["High"]), float(row["Low"])
    o = float(row["Open"])
    atr = float(row["ATR"]) if pd.notna(row.get("ATR")) else np.nan
    if pd.isna(atr) or atr <= 0 or _huge_candle(row, atr):
        return None

    pierce = 0.15 * atr
    rng = max(high - low, 1e-6)
    close_pos = (close - low) / rng

    # One PDH/PDL trade per day
    day = row["session_date"]
    prior = df.iloc[:i]
    same = prior[prior["session_date"] == day] if len(prior) else prior
    for _, prow in same.iterrows():
        if pd.isna(prow.get("PDH")):
            continue
        ph, pl = float(prow["PDH"]), float(prow["PDL"])
        if float(prow["High"]) > ph + pierce and float(prow["Close"]) < ph:
            return None
        if float(prow["Low"]) < pl - pierce and float(prow["Close"]) > pl:
            return None

    # Fade PDH reject → put  OR  continue break above PDH → call (prefer fade with confirmation)
    # Strict: wick beyond + close back (fade) OR clean close through with body (continue)
    # Use fade first (mean reversion at levels) — research both via mode param later; default fade
    if high > pdh + pierce and close < pdh and close < o and close_pos <= 0.40:
        return PatternSignal(
            pattern="pdh_pdl_killzone",
            side="put",
            confidence=0.75,
            reason=f"PDH reject fade @{pdh:.2f}",
            bar_time=ts,
            spot=close,
        )
    if low < pdl - pierce and close > pdl and close > o and close_pos >= 0.60:
        return PatternSignal(
            pattern="pdh_pdl_killzone",
            side="call",
            confidence=0.75,
            reason=f"PDL reclaim fade @{pdl:.2f}",
            bar_time=ts,
            spot=close,
        )
    # Continuation break (clean close beyond level)
    if close > pdh + 0.10 * atr and float(row["Low"]) <= pdh and close > o:
        return PatternSignal(
            pattern="pdh_pdl_killzone",
            side="call",
            confidence=0.73,
            reason=f"PDH break continue @{pdh:.2f}",
            bar_time=ts,
            spot=close,
        )
    if close < pdl - 0.10 * atr and float(row["High"]) >= pdl and close < o:
        return PatternSignal(
            pattern="pdh_pdl_killzone",
            side="put",
            confidence=0.73,
            reason=f"PDL break continue @{pdl:.2f}",
            bar_time=ts,
            spot=close,
        )
    return None


# ---------------------------------------------------------------------------
# 5) IB / OR failure — false break fade
# ---------------------------------------------------------------------------
def ib_failure_signal(df: pd.DataFrame, i: int) -> PatternSignal | None:
    row = df.iloc[i]
    ts = df.index[i]
    t = ts.time() if hasattr(ts, "time") else time(12, 0)
    if t < time(10, 5) or t > time(12, 0):
        return None
    if not bool(row.get("OR_READY", False)):
        return None

    same = session_slice(df, i)
    # Need a prior bar that broke OR, then this bar fails back inside
    oh, ol = float(row["OR_HIGH"]), float(row["OR_LOW"])
    close = float(row["Close"])
    high, low = float(row["High"]), float(row["Low"])
    atr = float(row["ATR"]) if pd.notna(row.get("ATR")) else np.nan
    if pd.isna(atr) or atr <= 0 or _huge_candle(row, atr):
        return None

    # Find first break earlier today
    side, break_j = first_orb_break(same.iloc[:-1] if len(same) > 1 else same.iloc[:0])
    if side is None or break_j is None:
        return None
    # Failure must occur within a few bars of the break (not late day)
    bars_since = len(same) - 1 - break_j
    if bars_since < 1 or bars_since > 8:
        return None

    # One IB failure per day
    prior_same = same.iloc[break_j + 1 : -1]
    for _, prow in prior_same.iterrows():
        pc = float(prow["Close"])
        if side == "call" and pc < oh and float(prow["High"]) > oh:
            return None
        if side == "put" and pc > ol and float(prow["Low"]) < ol:
            return None

    if side == "call":
        # Broke above, now closes back below OR high (failed breakout → put)
        if high > oh and close < oh and close < float(row["Open"]):
            return PatternSignal(
                pattern="ib_failure",
                side="put",
                confidence=0.77,
                reason=f"OR high failure fade @{oh:.2f}",
                bar_time=ts,
                spot=close,
            )
    else:
        if low < ol and close > ol and close > float(row["Open"]):
            return PatternSignal(
                pattern="ib_failure",
                side="call",
                confidence=0.77,
                reason=f"OR low failure fade @{ol:.2f}",
                bar_time=ts,
                spot=close,
            )
    return None
