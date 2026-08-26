"""Multi-timeframe liquidity: equal highs/lows on 1H/4H from 5m bars."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LiquidityLevel:
    price: float
    kind: str  # eqh | eql | swing_high | swing_low
    timeframe: str  # 1h | 4h
    touches: int
    last_ts: pd.Timestamp


def resample_ohlcv(df_5m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate RTH 5m into higher TF. Index = bar end (right-labeled)."""
    if df_5m.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    out = df_5m.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("America/New_York")
    agg = (
        out.resample(rule, label="right", closed="right")
        .agg(
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
            Volume=("Volume", "sum"),
        )
        .dropna(subset=["Open", "Close"])
    )
    return agg


def swing_points(
    df: pd.DataFrame,
    *,
    left: int = 2,
    right: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Fractal swing highs/lows. right>0 ⇒ confirmed only after `right` bars (no lookahead if caller truncates)."""
    high = df["High"]
    low = df["Low"]
    sh = pd.Series(np.nan, index=df.index, dtype=float)
    sl = pd.Series(np.nan, index=df.index, dtype=float)
    n = len(df)
    for i in range(left, n - right):
        window_h = high.iloc[i - left : i + right + 1]
        window_l = low.iloc[i - left : i + right + 1]
        if high.iloc[i] >= window_h.max():
            sh.iloc[i] = float(high.iloc[i])
        if low.iloc[i] <= window_l.min():
            sl.iloc[i] = float(low.iloc[i])
    return sh, sl


def _cluster_levels(
    points: list[tuple[pd.Timestamp, float]],
    *,
    tol_pct: float,
    kind: str,
    timeframe: str,
    min_touches: int = 2,
) -> list[LiquidityLevel]:
    if not points:
        return []
    points = sorted(points, key=lambda x: x[1])
    clusters: list[list[tuple[pd.Timestamp, float]]] = []
    for ts, px in points:
        placed = False
        for c in clusters:
            mid = float(np.mean([p for _, p in c]))
            if abs(px - mid) / max(mid, 1e-9) <= tol_pct:
                c.append((ts, px))
                placed = True
                break
        if not placed:
            clusters.append([(ts, px)])
    levels: list[LiquidityLevel] = []
    for c in clusters:
        if len(c) < min_touches:
            continue
        prices = [p for _, p in c]
        ts_last = max(t for t, _ in c)
        levels.append(
            LiquidityLevel(
                price=float(np.mean(prices)),
                kind=kind,
                timeframe=timeframe,
                touches=len(c),
                last_ts=ts_last,
            )
        )
    return levels


def equal_highs_lows(
    htf: pd.DataFrame,
    *,
    timeframe: str,
    tol_pct: float = 0.0015,
    swing_left: int = 2,
    swing_right: int = 2,
    lookback: int = 80,
    min_touches: int = 2,
) -> list[LiquidityLevel]:
    """EQH/EQL clusters from swing points on a completed HTF frame."""
    if htf is None or len(htf) < swing_left + swing_right + 3:
        return []
    frame = htf.iloc[-lookback:] if len(htf) > lookback else htf
    # Confirm swings only using bars inside `frame` (caller must pass completed bars only)
    sh, sl = swing_points(frame, left=swing_left, right=swing_right)
    highs = [(ts, float(v)) for ts, v in sh.dropna().items()]
    lows = [(ts, float(v)) for ts, v in sl.dropna().items()]
    eqh = _cluster_levels(highs, tol_pct=tol_pct, kind="eqh", timeframe=timeframe, min_touches=min_touches)
    eql = _cluster_levels(lows, tol_pct=tol_pct, kind="eql", timeframe=timeframe, min_touches=min_touches)
    # Also expose most recent raw swings as soft liquidity (1 touch)
    soft: list[LiquidityLevel] = []
    if highs:
        ts, px = highs[-1]
        soft.append(LiquidityLevel(px, "swing_high", timeframe, 1, ts))
    if lows:
        ts, px = lows[-1]
        soft.append(LiquidityLevel(px, "swing_low", timeframe, 1, ts))
    return eqh + eql + soft


def htf_structure_bias(htf_4h: pd.DataFrame) -> str:
    """
    Crude HTF bias from last completed 4H closes vs EMA21.
    Returns: bull | bear | neutral
    """
    if htf_4h is None or len(htf_4h) < 25:
        return "neutral"
    close = htf_4h["Close"]
    ema = close.ewm(span=21, adjust=False).mean()
    c, e = float(close.iloc[-1]), float(ema.iloc[-1])
    # Last two swing structure proxy: higher/lower closes
    c1, c2 = float(close.iloc[-2]), float(close.iloc[-1])
    if c > e * 1.001 and c2 >= c1:
        return "bull"
    if c < e * 0.999 and c2 <= c1:
        return "bear"
    return "neutral"


@dataclass
class MTFLiquidityMap:
    levels_1h: list[LiquidityLevel]
    levels_4h: list[LiquidityLevel]
    bias_4h: str

    def eq_levels(self, *, kind: str | None = None, min_touches: int = 2) -> list[LiquidityLevel]:
        out = [lv for lv in self.levels_1h + self.levels_4h if lv.touches >= min_touches]
        if kind:
            out = [lv for lv in out if lv.kind == kind]
        return out

    def nearest(
        self,
        spot: float,
        *,
        side: str,
        kinds: tuple[str, ...] = ("eqh", "eql"),
        max_dist_pct: float = 0.004,
        min_touches: int = 2,
        require_timeframes: tuple[str, ...] | None = None,
    ) -> LiquidityLevel | None:
        """Nearest liquidity above (side=high) or below (side=low)."""
        cands = [
            lv
            for lv in self.levels_1h + self.levels_4h
            if lv.kind in kinds and lv.touches >= min_touches
        ]
        if require_timeframes:
            cands = [lv for lv in cands if lv.timeframe in require_timeframes]
        best: LiquidityLevel | None = None
        best_dist = float("inf")
        for lv in cands:
            if side == "high" and lv.price <= spot:
                continue
            if side == "low" and lv.price >= spot:
                continue
            dist = abs(lv.price - spot) / max(spot, 1e-9)
            if dist > max_dist_pct:
                continue
            if dist < best_dist:
                best_dist = dist
                best = lv
        return best


# Optional full-series HTF frames set by backtest/research to avoid per-bar resample.
_PRE_H1: pd.DataFrame | None = None
_PRE_H4: pd.DataFrame | None = None
_MAP_CACHE: dict[tuple, MTFLiquidityMap] = {}


def set_htf_cache(df_5m: pd.DataFrame | None) -> None:
    """Precompute 1H/4H once for a 5m series (call before a backtest loop)."""
    global _PRE_H1, _PRE_H4, _MAP_CACHE
    _MAP_CACHE = {}
    if df_5m is None or df_5m.empty:
        _PRE_H1, _PRE_H4 = None, None
        return
    frame = df_5m
    if frame.index.tz is None:
        frame = frame.copy()
        frame.index = frame.index.tz_localize("America/New_York")
    _PRE_H1 = resample_ohlcv(frame, "1h")
    _PRE_H4 = resample_ohlcv(frame, "4h")


def clear_htf_cache() -> None:
    set_htf_cache(None)


def build_mtf_map(
    df_5m: pd.DataFrame,
    asof: pd.Timestamp,
    *,
    tol_pct: float = 0.0015,
    lookback: int = 80,
) -> MTFLiquidityMap:
    """
    Liquidity map using only HTF bars completed strictly before `asof`
    (no lookahead into the current incomplete hour/4H).
    """
    if asof.tzinfo is None:
        asof = asof.tz_localize("America/New_York")
    else:
        asof = asof.tz_convert("America/New_York")

    if _PRE_H1 is not None and _PRE_H4 is not None:
        h1_full, h4_full = _PRE_H1, _PRE_H4
    else:
        if df_5m.index.tz is None:
            df_5m = df_5m.copy()
            df_5m.index = df_5m.index.tz_localize("America/New_York")
        hist = df_5m[df_5m.index < asof]
        h1_full = resample_ohlcv(hist, "1h")
        h4_full = resample_ohlcv(hist, "4h")

    h1 = h1_full[h1_full.index < asof]
    h4 = h4_full[h4_full.index < asof]
    # Cache by last completed HTF stamps (levels only change when a new HTF bar completes)
    last1 = h1.index[-1] if len(h1) else pd.NaT
    last4 = h4.index[-1] if len(h4) else pd.NaT
    key = (last1, last4, round(float(tol_pct), 6), int(lookback))
    cached = _MAP_CACHE.get(key)
    if cached is not None:
        return cached
    m = MTFLiquidityMap(
        levels_1h=equal_highs_lows(h1, timeframe="1h", tol_pct=tol_pct, lookback=lookback),
        levels_4h=equal_highs_lows(h4, timeframe="4h", tol_pct=tol_pct, lookback=max(20, lookback // 4)),
        bias_4h=htf_structure_bias(h4),
    )
    _MAP_CACHE[key] = m
    return m


def open_volume_flood(df_5m: pd.DataFrame, i: int, *, lookback_days: int = 10) -> float:
    """
    Ratio of this bar's volume vs median first-hour volume over prior sessions.
    >1.0 ≈ above-average open participation.
    """
    if i < 1 or df_5m.empty:
        return 1.0
    row = df_5m.iloc[i]
    vol = float(row["Volume"])
    day = df_5m.index[i].date()
    # Prior RTH opens 09:30–10:30 volumes
    prior = df_5m.iloc[:i]
    if prior.empty:
        return 1.0
    times = prior.index
    mask = []
    for ts in times:
        t = ts.time()
        if ts.date() != day and t.hour == 9 and 30 <= t.minute <= 55:
            mask.append(True)
        elif ts.date() != day and t.hour == 10 and t.minute <= 30:
            mask.append(True)
        else:
            mask.append(False)
    base = prior.loc[mask, "Volume"] if any(mask) else prior["Volume"]
    if len(base) < 5:
        return 1.0
    # Limit to recent sessions
    med = float(base.tail(lookback_days * 12).median())
    if med <= 0:
        return 1.0
    return vol / med
