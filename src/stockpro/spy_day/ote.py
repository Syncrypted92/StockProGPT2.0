"""Optimal Trade Entry (OTE): Fib 61.8–78.6 retrace with optional OB/FVG confluence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Foxstars / ICT-style OTE zone
OTE_START = 0.618
OTE_SWEET = 0.705
OTE_END = 0.786


@dataclass(frozen=True)
class ImpulseSwing:
    direction: str  # bull | bear
    start_idx: int
    end_idx: int
    start_price: float
    end_price: float
    bos: bool

    @property
    def range(self) -> float:
        return abs(self.end_price - self.start_price)

    def fib_level(self, pct: float) -> float:
        """Retracement price at pct (0=end of impulse, 1=start)."""
        if self.direction == "bull":
            # low -> high; retrace down toward low
            return self.end_price - pct * (self.end_price - self.start_price)
        # high -> low; retrace up toward high
        return self.end_price + pct * (self.start_price - self.end_price)

    def ote_bounds(self) -> tuple[float, float, float]:
        """Return (level_618, level_705, level_786) as prices."""
        return self.fib_level(OTE_START), self.fib_level(OTE_SWEET), self.fib_level(OTE_END)

    def in_ote_zone(self, price: float) -> bool:
        a, _, b = self.ote_bounds()
        lo, hi = min(a, b), max(a, b)
        return lo <= price <= hi


@dataclass(frozen=True)
class OrderBlock:
    side: str  # bull | bear
    low: float
    high: float
    idx: int

    def overlaps_zone(self, z_lo: float, z_hi: float) -> bool:
        return not (self.high < z_lo or self.low > z_hi)


@dataclass(frozen=True)
class FairValueGap:
    side: str  # bull | bear
    low: float
    high: float
    idx: int  # middle candle index

    def overlaps_zone(self, z_lo: float, z_hi: float) -> bool:
        return not (self.high < z_lo or self.low > z_hi)


def _swing_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> tuple[list[int], list[int]]:
    highs: list[int] = []
    lows: list[int] = []
    n = len(df)
    for i in range(left, n - right):
        window_h = df["High"].iloc[i - left : i + right + 1]
        window_l = df["Low"].iloc[i - left : i + right + 1]
        if float(df["High"].iloc[i]) >= float(window_h.max()):
            highs.append(i)
        if float(df["Low"].iloc[i]) <= float(window_l.min()):
            lows.append(i)
    return highs, lows


# Per-bar impulse cache (built once per backtest/scan session)
_IMPULSE_CACHE: dict[int, ImpulseSwing | None] = {}


def clear_ote_cache() -> None:
    _IMPULSE_CACHE.clear()


def build_ote_cache(df: pd.DataFrame, *, window_start=None, window_end=None) -> None:
    """Precompute impulses for bars in the OTE entry window (speeds backtests)."""
    from datetime import time as dt_time

    clear_ote_cache()
    ws = window_start or dt_time(10, 0)
    we = window_end or dt_time(14, 30)
    for i in range(len(df)):
        ts = df.index[i]
        t = ts.time() if hasattr(ts, "time") else dt_time(12, 0)
        if t < ws or t > we:
            continue
        _IMPULSE_CACHE[i] = _find_impulse_uncached(df, i)


def find_impulse(
    df: pd.DataFrame,
    asof_i: int,
    *,
    lookback: int = 36,
    min_atr_mult: float = 1.8,
    pivot_left: int = 2,
    pivot_right: int = 2,
) -> ImpulseSwing | None:
    """Most recent confirmed impulse ending before asof_i."""
    if asof_i in _IMPULSE_CACHE:
        return _IMPULSE_CACHE[asof_i]
    return _find_impulse_uncached(df, asof_i)


def _find_impulse_uncached(
    df: pd.DataFrame,
    asof_i: int,
    *,
    lookback: int = 36,
    min_atr_mult: float = 1.8,
    pivot_left: int = 2,
    pivot_right: int = 2,
) -> ImpulseSwing | None:
    """
    Most recent confirmed impulse ending before asof_i.
    Bull: swing low -> swing high with optional BOS (close above prior swing high).
    Bear: swing high -> swing low with optional BOS (close below prior swing low).
    """
    if asof_i < pivot_left + pivot_right + 5:
        return None
    start = max(0, asof_i - lookback)
    # Only use bars up to asof_i - 1 so impulse is complete before entry bar
    frame = df.iloc[start:asof_i]
    if len(frame) < pivot_left + pivot_right + 5:
        return None
    atr = float(df["ATR"].iloc[asof_i]) if "ATR" in df.columns and pd.notna(df["ATR"].iloc[asof_i]) else np.nan
    if pd.isna(atr) or atr <= 0:
        atr = float((frame["High"] - frame["Low"]).median())

    highs, lows = _swing_pivots(frame, left=pivot_left, right=pivot_right)
    if not highs or not lows:
        return None

    # Map frame-local indices to global
    def g(local: int) -> int:
        return start + local

    candidates: list[ImpulseSwing] = []

    # Bull: most recent swing high + latest swing low before it (not all pairs)
    for hi in reversed(highs):
        lows_before = [li for li in lows if li < hi]
        if not lows_before:
            continue
        li = lows_before[-1]
        lo_px = float(frame["Low"].iloc[li])
        hi_px = float(frame["High"].iloc[hi])
        rng = hi_px - lo_px
        if rng < min_atr_mult * atr:
            continue
        prior_hs = [float(frame["High"].iloc[h]) for h in highs if h < li]
        bos = False
        if prior_hs:
            level = max(prior_hs)
            seg = frame.iloc[li : hi + 1]
            bos = bool((seg["Close"] > level).any())
        candidates.append(ImpulseSwing("bull", g(li), g(hi), lo_px, hi_px, bos))
        break

    # Bear: most recent swing low + latest swing high before it
    for li in reversed(lows):
        highs_before = [hi for hi in highs if hi < li]
        if not highs_before:
            continue
        hi = highs_before[-1]
        hi_px = float(frame["High"].iloc[hi])
        lo_px = float(frame["Low"].iloc[li])
        rng = hi_px - lo_px
        if rng < min_atr_mult * atr:
            continue
        prior_ls = [float(frame["Low"].iloc[x]) for x in lows if x < hi]
        bos = False
        if prior_ls:
            level = min(prior_ls)
            seg = frame.iloc[hi : li + 1]
            bos = bool((seg["Close"] < level).any())
        candidates.append(ImpulseSwing("bear", g(hi), g(li), hi_px, lo_px, bos))
        break

    if not candidates:
        return None
    # Prefer bull/bear with BOS if both exist; else most recent end
    candidates.sort(key=lambda s: (s.end_idx, int(s.bos), s.range), reverse=True)
    return candidates[0]


def find_order_block(df: pd.DataFrame, impulse: ImpulseSwing) -> OrderBlock | None:
    """
    Standard ICT OB (more reliable than social-media wording):
    - Bull impulse: last bearish candle before/at the impulse start (demand).
    - Bear impulse: last bullish candle before/at the impulse start (supply).
    """
    i0 = impulse.start_idx
    search_from = max(0, i0 - 8)
    if impulse.direction == "bull":
        for j in range(i0, search_from - 1, -1):
            row = df.iloc[j]
            if float(row["Close"]) < float(row["Open"]):
                return OrderBlock("bull", float(row["Low"]), float(row["High"]), j)
    else:
        for j in range(i0, search_from - 1, -1):
            row = df.iloc[j]
            if float(row["Close"]) > float(row["Open"]):
                return OrderBlock("bear", float(row["Low"]), float(row["High"]), j)
    return None


def find_fvgs_in_impulse(df: pd.DataFrame, impulse: ImpulseSwing) -> list[FairValueGap]:
    """3-candle FVGs created during the impulse leg."""
    out: list[FairValueGap] = []
    a, b = min(impulse.start_idx, impulse.end_idx), max(impulse.start_idx, impulse.end_idx)
    for i in range(max(a, 1), min(b, len(df) - 2)):
        c0 = df.iloc[i - 1]
        c2 = df.iloc[i + 1]
        # Bullish FVG: gap up (c0.high < c2.low)
        if float(c0["High"]) < float(c2["Low"]):
            out.append(
                FairValueGap("bull", float(c0["High"]), float(c2["Low"]), i)
            )
        # Bearish FVG: gap down (c0.low > c2.high)
        if float(c0["Low"]) > float(c2["High"]):
            out.append(
                FairValueGap("bear", float(c2["High"]), float(c0["Low"]), i)
            )
    return out


def confirmation_candle(df: pd.DataFrame, i: int, direction: str) -> bool:
    """Strong close in trade direction (avoid blind OTE touch)."""
    row = df.iloc[i]
    o, c = float(row["Open"]), float(row["Close"])
    h, l = float(row["High"]), float(row["Low"])
    rng = max(h - l, 1e-9)
    body = abs(c - o)
    if direction == "bull":
        return c > o and body / rng >= 0.45 and (c - l) / rng >= 0.60
    return c < o and body / rng >= 0.45 and (h - c) / rng >= 0.60


@dataclass
class OteSetup:
    impulse: ImpulseSwing
    ob: OrderBlock | None
    fvg: FairValueGap | None
    grade: str  # good | better | best
    zone_lo: float
    zone_hi: float


def evaluate_ote_at(
    df: pd.DataFrame,
    i: int,
    *,
    require_bos: bool = False,
    require_ob: bool = False,
    min_grade: str = "good",
) -> OteSetup | None:
    """Build OTE setup context at bar i (impulse must already be complete)."""
    impulse = find_impulse(df, i)
    if impulse is None:
        return None
    if require_bos and not impulse.bos:
        return None
    # Need some room after impulse end before entry
    if i <= impulse.end_idx + 1:
        return None

    z618, z705, z786 = impulse.ote_bounds()
    z_lo, z_hi = min(z618, z786), max(z618, z786)

    ob = find_order_block(df, impulse)
    fvgs = [
        f
        for f in find_fvgs_in_impulse(df, impulse)
        if f.side == impulse.direction and f.overlaps_zone(z_lo, z_hi)
    ]
    fvg = fvgs[-1] if fvgs else None

    ob_ok = ob is not None and ob.overlaps_zone(z_lo, z_hi)
    if require_ob and not ob_ok:
        return None

    if ob_ok and fvg is not None:
        grade = "best"
    elif ob_ok:
        grade = "better"
    else:
        grade = "good"

    rank = {"good": 0, "better": 1, "best": 2}
    if rank[grade] < rank.get(min_grade, 0):
        return None

    return OteSetup(impulse=impulse, ob=ob if ob_ok else None, fvg=fvg, grade=grade, zone_lo=z_lo, zone_hi=z_hi)


def price_in_ote(setup: OteSetup, low: float, high: float) -> bool:
    """True if bar range tags the OTE zone."""
    return not (high < setup.zone_lo or low > setup.zone_hi)
