"""Elite rule-based 5m SPY pattern detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd


@dataclass
class PatternSignal:
    pattern: str
    side: str  # call | put
    confidence: float
    reason: str
    bar_time: pd.Timestamp
    spot: float


def enrich_bars(df: pd.DataFrame, *, orb_minutes: int = 30) -> pd.DataFrame:
    """Add VWAP, EMAs, ATR, prior-day H/L, opening-range levels."""
    out = df.copy()
    if out.empty:
        return out
    idx = out.index
    if idx.tz is None:
        out.index = idx.tz_localize("America/New_York")
    dates = out.index.date
    # Session VWAP
    typical = (out["High"] + out["Low"] + out["Close"]) / 3.0
    pv = typical * out["Volume"]
    day = pd.Series(dates, index=out.index)
    cum_pv = pv.groupby(day).cumsum()
    cum_vol = out["Volume"].groupby(day).cumsum().replace(0, np.nan)
    out["VWAP"] = cum_pv / cum_vol
    out["EMA9"] = out["Close"].ewm(span=9, adjust=False).mean()
    out["EMA21"] = out["Close"].ewm(span=21, adjust=False).mean()
    # ATR(14) on 5m
    prev_close = out["Close"].shift(1)
    tr = pd.concat(
        [
            (out["High"] - out["Low"]).abs(),
            (out["High"] - prev_close).abs(),
            (out["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["ATR"] = tr.rolling(14).mean()
    out["ATR_pct"] = out["ATR"] / out["Close"]
    # Prior day high/low
    daily_hi = out["High"].groupby(day).max()
    daily_lo = out["Low"].groupby(day).min()
    prev_hi = daily_hi.shift(1)
    prev_lo = daily_lo.shift(1)
    out["PDH"] = day.map(prev_hi)
    out["PDL"] = day.map(prev_lo)
    # Opening range
    orb_bars = max(1, int(orb_minutes // 5))
    or_high = []
    or_low = []
    for d, g in out.groupby(day):
        g = g.sort_index()
        window = g.iloc[:orb_bars]
        oh = float(window["High"].max()) if len(window) else np.nan
        ol = float(window["Low"].min()) if len(window) else np.nan
        or_high.extend([oh] * len(g))
        or_low.extend([ol] * len(g))
    out["OR_HIGH"] = or_high
    out["OR_LOW"] = or_low
    bar_in_day = out.groupby(day).cumcount()
    out["OR_READY"] = bar_in_day >= orb_bars
    out["bar_in_day"] = bar_in_day
    out["session_date"] = day
    # Range expansion filter helpers
    out["range_pct"] = (out["High"] - out["Low"]) / out["Close"]
    out["below_vwap"] = out["Close"] < out["VWAP"]
    out["above_vwap"] = out["Close"] > out["VWAP"]
    # Overnight gap: session open vs prior session close
    opens = out["Open"].groupby(day).transform("first")
    day_close = out["Close"].groupby(day).last()
    prev_close_map = day_close.shift(1)
    prior_c = day.map(prev_close_map)
    out["GAP_PCT"] = (opens - prior_c) / prior_c.replace(0, np.nan)
    return out


def _huge_candle(row: pd.Series, atr: float) -> bool:
    if pd.isna(atr) or atr <= 0:
        return False
    return float(row["High"] - row["Low"]) > 2.5 * atr


# Set by backtest/scan from SpyDayConfig.htf_permission
HTF_PERMISSION_CFG = None  # HtfPermissionConfig | None


def configure_htf_permission(cfg) -> None:
    """cfg: HtfPermissionConfig | dict | None"""
    global HTF_PERMISSION_CFG
    from stockpro.spy_day.htf_permission import HtfPermissionConfig

    if cfg is None:
        HTF_PERMISSION_CFG = HtfPermissionConfig()
    elif isinstance(cfg, HtfPermissionConfig):
        HTF_PERMISSION_CFG = cfg
    else:
        HTF_PERMISSION_CFG = HtfPermissionConfig.from_dict(dict(cfg))


def _orb_signal(df: pd.DataFrame, i: int) -> PatternSignal | None:
    from stockpro.spy_day.htf_permission import HtfPermissionConfig, orb_htf_allowed

    row = df.iloc[i]
    if not bool(row.get("OR_READY", False)):
        return None
    # Only first break of the day
    day = row["session_date"]
    prior = df.iloc[:i]
    same = prior[prior["session_date"] == day]
    if len(same) and (
        (same["Close"] > same["OR_HIGH"]).any() or (same["Close"] < same["OR_LOW"]).any()
    ):
        # Already broke earlier — only allow if this bar is the first close beyond
        first_up = same[same["Close"] > same["OR_HIGH"]]
        first_dn = same[same["Close"] < same["OR_LOW"]]
        if len(first_up) or len(first_dn):
            return None
    oh, ol = float(row["OR_HIGH"]), float(row["OR_LOW"])
    close = float(row["Close"])
    atr = float(row["ATR"]) if pd.notna(row["ATR"]) else np.nan
    if _huge_candle(row, atr):
        return None
    ts = df.index[i]
    htf_cfg = HTF_PERMISSION_CFG or HtfPermissionConfig()

    if close > oh and float(row["Low"]) <= oh * 1.001:
        # breakout with some proximity (not far chase)
        if close > oh + 1.5 * (atr if pd.notna(atr) else 0):
            return None
        ok, note = orb_htf_allowed(df, i, "call", close, cfg=htf_cfg)
        if not ok:
            return None
        conf = 0.72
        if bool(row.get("above_vwap", False)):
            conf += 0.05
        if float(row["EMA9"]) > float(row["EMA21"]):
            conf += 0.05
        if note:
            conf = min(conf + 0.03, 0.95)
        reason = f"ORB break above {oh:.2f}"
        if note:
            reason = f"{reason} [{note}]"
        return PatternSignal(
            pattern="orb",
            side="call",
            confidence=min(conf, 0.95),
            reason=reason,
            bar_time=ts,
            spot=close,
        )
    if close < ol and float(row["High"]) >= ol * 0.999:
        if close < ol - 1.5 * (atr if pd.notna(atr) else 0):
            return None
        ok, note = orb_htf_allowed(df, i, "put", close, cfg=htf_cfg)
        if not ok:
            return None
        conf = 0.72
        if bool(row.get("below_vwap", False)):
            conf += 0.05
        if float(row["EMA9"]) < float(row["EMA21"]):
            conf += 0.05
        if note:
            conf = min(conf + 0.03, 0.95)
        reason = f"ORB break below {ol:.2f}"
        if note:
            reason = f"{reason} [{note}]"
        return PatternSignal(
            pattern="orb",
            side="put",
            confidence=min(conf, 0.95),
            reason=reason,
            bar_time=ts,
            spot=close,
        )
    return None


def _vwap_signal(df: pd.DataFrame, i: int) -> PatternSignal | None:
    if i < 3:
        return None
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    prev2 = df.iloc[i - 2]
    close = float(row["Close"])
    vwap = float(row["VWAP"]) if pd.notna(row["VWAP"]) else np.nan
    if pd.isna(vwap):
        return None
    atr = float(row["ATR"]) if pd.notna(row["ATR"]) else np.nan
    if _huge_candle(row, atr):
        return None
    ts = df.index[i]
    # Reclaim: was below, now closes above with EMA support
    if (
        bool(prev2.get("below_vwap", False))
        and bool(prev.get("below_vwap", False))
        and close > vwap
        and float(row["EMA9"]) >= float(row["EMA21"]) * 0.999
    ):
        return PatternSignal(
            pattern="vwap_reclaim",
            side="call",
            confidence=0.70,
            reason=f"VWAP reclaim {vwap:.2f}",
            bar_time=ts,
            spot=close,
        )
    if (
        bool(prev2.get("above_vwap", False))
        and bool(prev.get("above_vwap", False))
        and close < vwap
        and float(row["EMA9"]) <= float(row["EMA21"]) * 1.001
    ):
        return PatternSignal(
            pattern="vwap_reclaim",
            side="put",
            confidence=0.70,
            reason=f"VWAP reject {vwap:.2f}",
            bar_time=ts,
            spot=close,
        )
    return None


def _ema_pullback_signal(df: pd.DataFrame, i: int) -> PatternSignal | None:
    if i < 25:
        return None
    row = df.iloc[i]
    close = float(row["Close"])
    e9, e21 = float(row["EMA9"]), float(row["EMA21"])
    atr = float(row["ATR"]) if pd.notna(row["ATR"]) else np.nan
    if pd.isna(atr) or atr <= 0:
        return None
    if _huge_candle(row, atr):
        return None
    ts = df.index[i]
    # Bull stack: pullback into EMA9 then green close
    if e9 > e21 and close > e9:
        touched = float(row["Low"]) <= e9 + 0.15 * atr
        bullish_bar = close > float(row["Open"])
        if touched and bullish_bar and bool(row.get("above_vwap", False)):
            return PatternSignal(
                pattern="ema_pullback",
                side="call",
                confidence=0.68,
                reason="EMA9 pullback resume long",
                bar_time=ts,
                spot=close,
            )
    if e9 < e21 and close < e9:
        touched = float(row["High"]) >= e9 - 0.15 * atr
        bearish_bar = close < float(row["Open"])
        if touched and bearish_bar and bool(row.get("below_vwap", False)):
            return PatternSignal(
                pattern="ema_pullback",
                side="put",
                confidence=0.68,
                reason="EMA9 pullback resume short",
                bar_time=ts,
                spot=close,
            )
    return None


# Tunables for open MTF liquidity sweep (research script may override).
# Defaults = best sweep-only research variant (open_wide_eq); still below gate alone.
SWEEP_PARAMS: dict = {
    "window_start": time(9, 35),
    "window_end": time(10, 30),
    "eq_tol_pct": 0.0025,
    "max_level_dist_pct": 0.005,
    "pierce_atr": 0.10,
    "min_touches": 2,
    "require_eq": True,
    "require_4h_level": False,
    "require_dual_tf": False,
    "bias_mode": "fade_with_htf",
    "min_vol_flood": 0.70,
    "allow_pdh_pdl": True,
    "one_per_day": True,
}


def _pdh_pdl_signal(df: pd.DataFrame, i: int) -> PatternSignal | None:
    """Legacy alias — prefer liquidity_sweep."""
    return _liquidity_sweep_signal(df, i)


def _already_swept_today(df: pd.DataFrame, i: int) -> bool:
    row = df.iloc[i]
    day = row["session_date"]
    prior = df.iloc[:i]
    if prior.empty:
        return False
    same = prior[prior["session_date"] == day]
    # Heuristic: any prior morning reject wick beyond PDH/PDL or tagged sweep reason not stored —
    # use a soft flag: prior bar already pierced PDH/PDL and closed back.
    for _, prow in same.iterrows():
        pdh, pdl = prow.get("PDH"), prow.get("PDL")
        if pd.isna(pdh) or pd.isna(pdl):
            continue
        pdh, pdl = float(pdh), float(pdl)
        if float(prow["High"]) > pdh and float(prow["Close"]) < pdh:
            return True
        if float(prow["Low"]) < pdl and float(prow["Close"]) > pdl:
            return True
    return False


def _liquidity_sweep_signal(df: pd.DataFrame, i: int) -> PatternSignal | None:
    """
    Open liquidity sweep (new money flood):

    Detect on 5m — wick through resting liquidity (equal highs/lows from 1H/4H,
    else PDH/PDL), close back inside. Verify bias on 4H. Window is early session
    when volume typically expands (not the later OR-only fade).
    """
    from stockpro.spy_day.mtf_liquidity import build_mtf_map, open_volume_flood

    p = SWEEP_PARAMS
    row = df.iloc[i]
    ts = df.index[i]
    t = ts.time() if hasattr(ts, "time") else time(12, 0)
    if t < p["window_start"] or t > p["window_end"]:
        return None

    close = float(row["Close"])
    high = float(row["High"])
    low = float(row["Low"])
    o = float(row["Open"])
    atr = float(row["ATR"]) if pd.notna(row["ATR"]) else np.nan
    if pd.isna(atr) or atr <= 0:
        return None
    if _huge_candle(row, atr):
        return None

    if p.get("one_per_day") and _already_swept_today(df, i):
        return None

    flood = open_volume_flood(df, i)
    if flood < float(p.get("min_vol_flood", 0.0)):
        return None

    mtf = build_mtf_map(df, ts, tol_pct=float(p["eq_tol_pct"]))
    bias = mtf.bias_4h
    pierce = float(p["pierce_atr"]) * atr
    rng = max(high - low, 1e-6)
    close_pos = (close - low) / rng
    max_dist = float(p["max_level_dist_pct"])
    min_touches = int(p["min_touches"])

    # Dual-TF: need EQH (or EQL) on both frames within band of each other
    def _dual_ok(side: str) -> bool:
        if not p.get("require_dual_tf"):
            return True
        kinds = ("eqh",) if side == "high" else ("eql",)
        a = [lv for lv in mtf.levels_1h if lv.kind in kinds and lv.touches >= min_touches]
        b = [lv for lv in mtf.levels_4h if lv.kind in kinds and lv.touches >= min_touches]
        for x in a:
            for y in b:
                if abs(x.price - y.price) / max(x.price, 1e-9) <= float(p["eq_tol_pct"]) * 2:
                    return True
        return False

    def _pick_level(side: str):
        kinds = ("eqh",) if side == "high" else ("eql",)
        req_tf = ("4h",) if p.get("require_4h_level") else None
        lv = mtf.nearest(
            close,
            side=side,
            kinds=kinds,
            max_dist_pct=max_dist,
            min_touches=min_touches,
            require_timeframes=req_tf,
        )
        if lv is not None:
            return lv, "eq"
        if p.get("allow_pdh_pdl") and not p.get("require_eq"):
            pdh, pdl = row.get("PDH"), row.get("PDL")
            if side == "high" and pd.notna(pdh):
                dist = abs(float(pdh) - close) / close
                if dist <= max_dist and float(pdh) >= close:
                    return type("L", (), {"price": float(pdh), "kind": "pdh", "timeframe": "1d", "touches": 1})(), "pd"
            if side == "low" and pd.notna(pdl):
                dist = abs(float(pdl) - close) / close
                if dist <= max_dist and float(pdl) <= close:
                    return type("L", (), {"price": float(pdl), "kind": "pdl", "timeframe": "1d", "touches": 1})(), "pd"
        if p.get("allow_pdh_pdl") and p.get("require_eq"):
            # Fallback only when EQ missing but PDH/PDL is the open magnet
            pdh, pdl = row.get("PDH"), row.get("PDL")
            if side == "high" and pd.notna(pdh):
                dist = abs(float(pdh) - close) / close
                if dist <= max_dist * 1.2 and float(pdh) >= close * 0.999:
                    return type("L", (), {"price": float(pdh), "kind": "pdh", "timeframe": "1d", "touches": 1})(), "pd"
            if side == "low" and pd.notna(pdl):
                dist = abs(float(pdl) - close) / close
                if dist <= max_dist * 1.2 and float(pdl) <= close * 1.001:
                    return type("L", (), {"price": float(pdl), "kind": "pdl", "timeframe": "1d", "touches": 1})(), "pd"
        return None, None

    def _bias_allows(side_trade: str) -> bool:
        mode = str(p.get("bias_mode", "none"))
        if mode == "none":
            return True
        # fade_with_htf: put sweeps preferred when HTF not strongly bull; calls when not strongly bear
        if mode == "fade_with_htf":
            if side_trade == "put" and bias == "bull":
                return False
            if side_trade == "call" and bias == "bear":
                return False
            return True
        # fade_only: only fade into HTF premium/discount (put into bull HTF / call into bear)
        if mode == "fade_only":
            if side_trade == "put" and bias == "bull":
                return True
            if side_trade == "call" and bias == "bear":
                return True
            return bias == "neutral"
        return True

    # --- Sweep highs (EQH / PDH) → put ---
    lv_hi, src_hi = _pick_level("high")
    if lv_hi is not None and _dual_ok("high") and _bias_allows("put"):
        lvl = float(lv_hi.price)
        if high > lvl + pierce and close < lvl and close < o and close_pos <= 0.45:
            conf = 0.78
            if getattr(lv_hi, "kind", "") in ("eqh",) and int(getattr(lv_hi, "touches", 1)) >= 2:
                conf += 0.04
            if getattr(lv_hi, "timeframe", "") == "4h":
                conf += 0.03
            if flood >= 1.15:
                conf += 0.03
            if bias == "bear":
                conf += 0.02
            return PatternSignal(
                pattern="liquidity_sweep",
                side="put",
                confidence=min(conf, 0.93),
                reason=(
                    f"Open sweep {getattr(lv_hi, 'kind', 'hi')}@{lvl:.2f} "
                    f"({getattr(lv_hi, 'timeframe', '?')}, flood={flood:.2f}, 4h={bias})"
                ),
                bar_time=ts,
                spot=close,
            )

    # --- Sweep lows (EQL / PDL) → call ---
    lv_lo, src_lo = _pick_level("low")
    if lv_lo is not None and _dual_ok("low") and _bias_allows("call"):
        lvl = float(lv_lo.price)
        if low < lvl - pierce and close > lvl and close > o and close_pos >= 0.55:
            conf = 0.78
            if getattr(lv_lo, "kind", "") in ("eql",) and int(getattr(lv_lo, "touches", 1)) >= 2:
                conf += 0.04
            if getattr(lv_lo, "timeframe", "") == "4h":
                conf += 0.03
            if flood >= 1.15:
                conf += 0.03
            if bias == "bull":
                conf += 0.02
            return PatternSignal(
                pattern="liquidity_sweep",
                side="call",
                confidence=min(conf, 0.93),
                reason=(
                    f"Open sweep {getattr(lv_lo, 'kind', 'lo')}@{lvl:.2f} "
                    f"({getattr(lv_lo, 'timeframe', '?')}, flood={flood:.2f}, 4h={bias})"
                ),
                bar_time=ts,
                spot=close,
            )
    return None


# OTE tunables (research may override)
OTE_PARAMS: dict = {
    "window_start": time(10, 0),
    "window_end": time(14, 0),
    "require_bos": False,
    "require_ob": False,
    "min_grade": "good",  # good | better | best
    "one_per_day": True,
    "require_confirmation": True,
    "killzone_only": True,  # NY morning + early afternoon
}

_OTE_FIRED_DAYS: set = set()


def clear_ote_day_state() -> None:
    _OTE_FIRED_DAYS.clear()


def configure_ote_params(raw: dict | None = None) -> None:
    """Apply spy_day.ote settings without touching ORB / HTF permission."""
    if not raw:
        return
    if "min_grade" in raw:
        OTE_PARAMS["min_grade"] = str(raw["min_grade"])
    if "require_bos" in raw:
        OTE_PARAMS["require_bos"] = bool(raw["require_bos"])
    if "require_ob" in raw:
        OTE_PARAMS["require_ob"] = bool(raw["require_ob"])
    if "require_confirmation" in raw:
        OTE_PARAMS["require_confirmation"] = bool(raw["require_confirmation"])
    if "killzone_only" in raw:
        OTE_PARAMS["killzone_only"] = bool(raw["killzone_only"])
    if "one_per_day" in raw:
        OTE_PARAMS["one_per_day"] = bool(raw["one_per_day"])


def _ote_signal(df: pd.DataFrame, i: int) -> PatternSignal | None:
    """
    Optimal Trade Entry: after impulse, price retraces into Fib 61.8–78.6,
    preferably into OB/FVG confluence, with confirmation candle (not blind touch).
    """
    from stockpro.spy_day.ote import confirmation_candle, evaluate_ote_at, price_in_ote

    p = OTE_PARAMS
    ts = df.index[i]
    t = ts.time() if hasattr(ts, "time") else time(12, 0)
    if t < p["window_start"] or t > p["window_end"]:
        return None
    if p.get("killzone_only"):
        morning = time(10, 0) <= t <= time(11, 30)
        lunch = time(13, 30) <= t <= time(14, 0)
        if not (morning or lunch):
            return None

    row = df.iloc[i]
    day = row["session_date"]
    if p.get("one_per_day") and day in _OTE_FIRED_DAYS:
        return None

    atr = float(row["ATR"]) if pd.notna(row.get("ATR")) else np.nan
    if pd.isna(atr) or atr <= 0:
        return None
    if _huge_candle(row, atr):
        return None

    setup = evaluate_ote_at(
        df,
        i,
        require_bos=bool(p.get("require_bos", False)),
        require_ob=bool(p.get("require_ob", False)),
        min_grade=str(p.get("min_grade", "good")),
    )
    if setup is None:
        return None

    low, high = float(row["Low"]), float(row["High"])
    if not price_in_ote(setup, low, high):
        return None

    direction = setup.impulse.direction
    if p.get("require_confirmation", True) and not confirmation_candle(df, i, direction):
        return None

    side = "call" if direction == "bull" else "put"
    conf = 0.72
    if setup.grade == "better":
        conf += 0.06
    elif setup.grade == "best":
        conf += 0.10
    if setup.impulse.bos:
        conf += 0.03
    close = float(row["Close"])
    sweet = setup.impulse.fib_level(0.705)
    dist = abs(close - sweet) / max(close, 1e-9)
    if dist <= 0.0015:
        conf += 0.02

    bits = [f"OTE {setup.grade}", f"zone={setup.zone_lo:.2f}-{setup.zone_hi:.2f}"]
    if setup.ob:
        bits.append("OB")
    if setup.fvg:
        bits.append("FVG")
    if setup.impulse.bos:
        bits.append("BOS")

    if p.get("one_per_day"):
        _OTE_FIRED_DAYS.add(day)

    return PatternSignal(
        pattern="ote",
        side=side,
        confidence=min(conf, 0.95),
        reason=" ".join(bits),
        bar_time=ts,
        spot=close,
    )


_DETECTORS = {
    "orb": _orb_signal,
    "vwap_reclaim": _vwap_signal,
    "ema_pullback": _ema_pullback_signal,
    "pdh_pdl_sweep": _pdh_pdl_signal,
    "liquidity_sweep": _liquidity_sweep_signal,
    "ote": _ote_signal,
}

# Register secondary / intraday patterns (lazy import avoids circular init issues)
def _register_secondary() -> None:
    from stockpro.spy_day import amd as amd_mod
    from stockpro.spy_day import intraday_patterns as intra
    from stockpro.spy_day import secondary_patterns as sec

    _DETECTORS.update(
        {
            "orb_gap_align": sec.orb_gap_align_signal,
            "orb_retest": sec.orb_retest_signal,
            "vwap_orb_bias": sec.vwap_orb_bias_signal,
            "pdh_pdl_killzone": sec.pdh_pdl_killzone_signal,
            "ib_failure": sec.ib_failure_signal,
            "amd": amd_mod.amd_signal,
            "gap_and_go": intra.gap_and_go_signal,
            "opening_drive": intra.opening_drive_signal,
            "trend_day_pullback": intra.trend_day_pullback_signal,
            "failed_auction": intra.failed_auction_signal,
            "range_day_fade": intra.range_day_fade_signal,
            "power_hour": intra.power_hour_signal,
            "multi_day_break": intra.multi_day_break_signal,
            "compression_break": intra.compression_break_signal,
            "spy_qqq_lead": intra.spy_qqq_lead_signal,
        }
    )


_register_secondary()


# Higher = preferred when two patterns fire on the same bar (ORB never loses to OTE).
PATTERN_PRIORITY: dict[str, int] = {
    "orb": 100,
    "orb_gap_align": 95,
    "spy_qqq_lead": 90,
    "gap_and_go": 88,
    "opening_drive": 85,
    "compression_break": 82,
    "orb_retest": 80,
    "amd": 78,
    "multi_day_break": 78,
    "ib_failure": 75,
    "failed_auction": 74,
    "trend_day_pullback": 72,
    "vwap_orb_bias": 70,
    "power_hour": 65,
    "pdh_pdl_killzone": 60,
    "range_day_fade": 50,
    "ote": 40,
    "liquidity_sweep": 30,
    "vwap_reclaim": 20,
    "ema_pullback": 20,
    "pdh_pdl_sweep": 20,
}


def detect_patterns_at(
    df: pd.DataFrame,
    i: int | None = None,
    *,
    enabled: list[str] | None = None,
    min_confidence: float = 0.0,
    pattern_min_confidence: dict[str, float] | None = None,
    pattern_priority: dict[str, int] | None = None,
) -> list[PatternSignal]:
    """Detect patterns on bar i (default: last bar). Assumes enrich_bars already applied."""
    if df.empty:
        return []
    if i is None:
        i = len(df) - 1
    if i < 0 or i >= len(df):
        return []
    ts = df.index[i]
    if hasattr(ts, "time") and ts.time() < time(9, 35):
        return []
    names = enabled or list(_DETECTORS.keys())
    pmc = pattern_min_confidence or {}
    prio = pattern_priority or PATTERN_PRIORITY
    out: list[PatternSignal] = []
    for name in names:
        fn = _DETECTORS.get(name)
        if not fn:
            continue
        sig = fn(df, i)
        if not sig:
            continue
        from stockpro.spy_day.pdh_filter import pdh_allows
        from stockpro.spy_day.po3 import po3_allows
        from stockpro.spy_day.smt import smt_allows

        ok_po3, po3_note = po3_allows(df, i, sig)
        if not ok_po3:
            continue
        ok_smt, smt_note = smt_allows(df, i, sig)
        if not ok_smt:
            continue
        ok_pdh, pdh_note = pdh_allows(df, i, sig)
        if not ok_pdh:
            continue
        need = float(pmc.get(name, min_confidence))
        if sig.confidence >= need:
            extra = " ".join(
                n
                for n in (po3_note, smt_note, pdh_note)
                if n and (n.startswith("po3_judas") or n.startswith("smt_bull") or n.startswith("smt_bear") or n.startswith("pdh_"))
            )
            if extra:
                bump = 0.02 if "po3_judas" in extra or "smt_" in extra else 0.01
                sig = PatternSignal(
                    pattern=sig.pattern,
                    side=sig.side,
                    confidence=min(sig.confidence + bump, 0.95),
                    reason=f"{sig.reason} [{extra}]",
                    bar_time=sig.bar_time,
                    spot=sig.spot,
                )
            out.append(sig)
    # Priority first (ORB > OTE), then confidence — keeps strategies separate
    out.sort(key=lambda s: (int(prio.get(s.pattern, 0)), s.confidence), reverse=True)
    return out


def best_signal(
    df: pd.DataFrame,
    i: int | None = None,
    *,
    enabled: list[str] | None = None,
    min_confidence: float = 0.65,
    pattern_min_confidence: dict[str, float] | None = None,
    pattern_priority: dict[str, int] | None = None,
) -> PatternSignal | None:
    sigs = detect_patterns_at(
        df,
        i,
        enabled=enabled,
        min_confidence=min_confidence,
        pattern_min_confidence=pattern_min_confidence,
        pattern_priority=pattern_priority,
    )
    return sigs[0] if sigs else None
