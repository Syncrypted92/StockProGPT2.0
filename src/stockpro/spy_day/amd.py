"""AMD (Accumulation → Manipulation → Distribution) on SPY 5m.

Accumulation: first 30m opening range (same box as ORB).
Manipulation: liquidity sweep beyond OR that closes back inside.
Distribution: fade the sweep (classic AMD day model).

Live paper uses this with short-dated options (~3 DTE), not 0DTE.
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from stockpro.spy_day.patterns import PatternSignal, _huge_candle
from stockpro.spy_day.secondary_patterns import session_slice

# Tunables (research / live may override)
AMD_PARAMS: dict = {
    "window_start": time(10, 0),
    "window_end": time(12, 0),  # morning AMD
    "pierce_atr": 0.15,
    "one_per_day": True,
    "require_confirm": False,  # True = wait one bar after reclaim
    "apply_htf": True,  # 4H counter-trend block (tuned recipe)
    "min_body_frac": 0.35,  # reclaim candle should not be a doji
    "skip_friday": False,  # WF: Fri skip matched simple recipe; drop to reduce overfit
}


def configure_amd_params(raw: dict | None) -> None:
    if not raw:
        return
    for k, v in raw.items():
        if k not in AMD_PARAMS and k not in (
            "min_dte",
            "max_dte",
            "profit_target_pct",
            "stop_loss_pct",
            "enabled",
            "min_confidence",
        ):
            continue
        if k not in AMD_PARAMS:
            continue
        if k in ("window_start", "window_end") and isinstance(v, str):
            hh, mm = v.split(":")
            AMD_PARAMS[k] = time(int(hh), int(mm))
        else:
            AMD_PARAMS[k] = v


def _manipulation_on_bar(
    row: pd.Series,
    oh: float,
    ol: float,
    atr: float,
    pierce: float,
) -> str | None:
    """Return 'high_sweep' or 'low_sweep' if this bar manipulates then closes inside OR."""
    high, low, close, o = float(row["High"]), float(row["Low"]), float(row["Close"]), float(row["Open"])
    rng = max(high - low, 1e-6)
    body = abs(close - o)

    # High-side manipulation: take buys above OR, close back inside → distribution down (put)
    if high > oh + pierce and close < oh and close > ol:
        if body / rng < float(AMD_PARAMS["min_body_frac"]):
            return None
        if close > o:  # prefer rejection (bearish or at least not strong bull close)
            # allow weak; require close in lower 60% of range
            if (close - low) / rng > 0.60:
                return None
        return "high_sweep"

    # Low-side manipulation: take sells below OR, close back inside → distribution up (call)
    if low < ol - pierce and close > ol and close < oh:
        if body / rng < float(AMD_PARAMS["min_body_frac"]):
            return None
        if close < o:
            if (high - close) / rng > 0.60:
                return None
        return "low_sweep"
    return None


def _already_traded_amd(same: pd.DataFrame, oh: float, ol: float, atr: float) -> bool:
    if not AMD_PARAMS.get("one_per_day", True) or len(same) < 2:
        return False
    pierce = float(AMD_PARAMS["pierce_atr"]) * atr
    # Any prior reclaim in session counts as AMD already offered
    for _, prow in same.iloc[:-1].iterrows():
        if not bool(prow.get("OR_READY", False)):
            continue
        if _manipulation_on_bar(prow, oh, ol, atr, pierce):
            return True
    return False


def amd_signal(df: pd.DataFrame, i: int) -> PatternSignal | None:
    """AMD distribution entry after OR accumulation + manipulation sweep."""
    if i < 1:
        return None
    row = df.iloc[i]
    ts = df.index[i]
    t = ts.time() if hasattr(ts, "time") else time(12, 0)
    p = AMD_PARAMS
    if p.get("skip_friday") and hasattr(ts, "weekday") and ts.weekday() == 4:
        return None
    if t < p["window_start"] or t > p["window_end"]:
        return None
    if not bool(row.get("OR_READY", False)):
        return None

    oh, ol = float(row["OR_HIGH"]), float(row["OR_LOW"])
    if not (oh > ol > 0):
        return None
    atr = float(row["ATR"]) if pd.notna(row.get("ATR")) else np.nan
    if pd.isna(atr) or atr <= 0 or _huge_candle(row, atr):
        return None

    pierce = float(p["pierce_atr"]) * atr
    same = session_slice(df, i)
    if _already_traded_amd(same, oh, ol, atr):
        # If one_per_day and a prior manipulation already occurred, only allow
        # confirm-mode entry on the bar immediately after that first sweep.
        if not p.get("require_confirm"):
            return None

    close = float(row["Close"])
    side: str | None = None
    reason = ""

    if p.get("require_confirm"):
        prev = df.iloc[i - 1]
        if not bool(prev.get("OR_READY", False)):
            return None
        manip = _manipulation_on_bar(prev, float(prev["OR_HIGH"]), float(prev["OR_LOW"]), atr, pierce)
        if manip is None:
            return None
        # Confirm distribution candle
        if manip == "high_sweep":
            if close < float(prev["Close"]) and close < float(row["Open"]) and close < oh:
                side, reason = "put", f"AMD dist confirm after OR_H sweep @{oh:.2f}"
        elif manip == "low_sweep":
            if close > float(prev["Close"]) and close > float(row["Open"]) and close > ol:
                side, reason = "call", f"AMD dist confirm after OR_L sweep @{ol:.2f}"
        # one_per_day: ensure this is the first confirm opportunity
        if side and p.get("one_per_day"):
            # prior confirms already?
            prior = same.iloc[:-1]
            for j in range(1, len(prior)):
                prow = prior.iloc[j]
                prev2 = prior.iloc[j - 1]
                m2 = _manipulation_on_bar(
                    prev2,
                    float(prev2["OR_HIGH"]),
                    float(prev2["OR_LOW"]),
                    atr,
                    pierce,
                )
                if m2 and bool(prow.get("OR_READY", False)):
                    return None
    else:
        manip = _manipulation_on_bar(row, oh, ol, atr, pierce)
        if manip == "high_sweep":
            side, reason = "put", f"AMD manip OR_H sweep->dist put @{oh:.2f}"
        elif manip == "low_sweep":
            side, reason = "call", f"AMD manip OR_L sweep->dist call @{ol:.2f}"
        if side and p.get("one_per_day") and _already_traded_amd(same, oh, ol, atr):
            return None

    if side is None:
        return None

    if p.get("apply_htf"):
        from stockpro.spy_day.htf_permission import HtfPermissionConfig, orb_htf_allowed
        from stockpro.spy_day.patterns import HTF_PERMISSION_CFG

        htf_cfg = HTF_PERMISSION_CFG or HtfPermissionConfig()
        ok, note = orb_htf_allowed(df, i, side, close, cfg=htf_cfg)
        if not ok:
            return None
        if note:
            reason = f"{reason} [{note}]"

    conf = 0.76
    if abs(close - (oh + ol) / 2) / max(close, 1e-9) < 0.001:
        conf += 0.02  # closed near mid-range after sweep — cleaner
    return PatternSignal(
        pattern="amd",
        side=side,
        confidence=min(conf, 0.92),
        reason=reason,
        bar_time=ts,
        spot=close,
    )
