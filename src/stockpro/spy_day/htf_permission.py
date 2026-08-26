"""1H/4H permission filters for ORB triggers (not separate entries)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from stockpro.spy_day.mtf_liquidity import MTFLiquidityMap, build_mtf_map


@dataclass
class HtfPermissionConfig:
    enabled: bool = True
    # Block ORB calls into bearish 4H / puts into bullish 4H
    skip_4h_counter_trend: bool = True
    # EQ location: none | leave | toward | either
    # leave  = breakout leaving nearby EQ balance (best researched PF)
    # toward = breakout toward nearby EQ pool
    # either = leave OR toward
    eq_context: str = "leave"
    eq_tol_pct: float = 0.0025
    eq_max_dist_pct: float = 0.004
    eq_min_touches: int = 2

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> HtfPermissionConfig:
        raw = dict(raw or {})
        eq = str(raw.get("eq_context", "leave")).lower().strip()
        if eq not in {"none", "leave", "toward", "either"}:
            eq = "leave"
        return cls(
            enabled=bool(raw.get("enabled", True)),
            skip_4h_counter_trend=bool(raw.get("skip_4h_counter_trend", True)),
            eq_context=eq,
            eq_tol_pct=float(raw.get("eq_tol_pct", 0.0025)),
            eq_max_dist_pct=float(raw.get("eq_max_dist_pct", 0.004)),
            eq_min_touches=int(raw.get("eq_min_touches", 2)),
        )


def _counter_trend_blocked(side: str, bias_4h: str) -> bool:
    if side == "call" and bias_4h == "bear":
        return True
    if side == "put" and bias_4h == "bull":
        return True
    return False


def _eq_leave(mtf: MTFLiquidityMap, side: str, spot: float, *, max_dist: float, min_touches: int) -> bool:
    if side == "call":
        return (
            mtf.nearest(
                spot,
                side="low",
                kinds=("eql", "eqh"),
                max_dist_pct=max_dist,
                min_touches=min_touches,
            )
            is not None
        )
    return (
        mtf.nearest(
            spot,
            side="high",
            kinds=("eql", "eqh"),
            max_dist_pct=max_dist,
            min_touches=min_touches,
        )
        is not None
    )


def _eq_toward(mtf: MTFLiquidityMap, side: str, spot: float, *, max_dist: float, min_touches: int) -> bool:
    if side == "call":
        return (
            mtf.nearest(
                spot,
                side="high",
                kinds=("eqh",),
                max_dist_pct=max(max_dist, 0.006),
                min_touches=min_touches,
            )
            is not None
        )
    return (
        mtf.nearest(
            spot,
            side="low",
            kinds=("eql",),
            max_dist_pct=max(max_dist, 0.006),
            min_touches=min_touches,
        )
        is not None
    )


def orb_htf_allowed(
    df: pd.DataFrame,
    i: int,
    side: str,
    spot: float,
    *,
    cfg: HtfPermissionConfig | None = None,
) -> tuple[bool, str]:
    """
    Permission check for an ORB trigger.
    Returns (allowed, note) where note is appended to the signal reason when allowed.
    """
    cfg = cfg or HtfPermissionConfig()
    if not cfg.enabled:
        return True, ""

    ts = df.index[i]
    mtf = build_mtf_map(df, ts, tol_pct=cfg.eq_tol_pct)
    notes: list[str] = [f"4h={mtf.bias_4h}"]

    if cfg.skip_4h_counter_trend and _counter_trend_blocked(side, mtf.bias_4h):
        return False, f"blocked_counter_4h={mtf.bias_4h}"

    eq_mode = cfg.eq_context
    if eq_mode != "none":
        leave_ok = _eq_leave(
            mtf, side, spot, max_dist=cfg.eq_max_dist_pct, min_touches=cfg.eq_min_touches
        )
        toward_ok = _eq_toward(
            mtf, side, spot, max_dist=cfg.eq_max_dist_pct, min_touches=cfg.eq_min_touches
        )
        if eq_mode == "leave" and not leave_ok:
            return False, "blocked_no_eq_leave"
        if eq_mode == "toward" and not toward_ok:
            return False, "blocked_no_eq_toward"
        if eq_mode == "either" and not (leave_ok or toward_ok):
            return False, "blocked_no_eq_context"
        if leave_ok:
            notes.append("eq=leave")
        elif toward_ok:
            notes.append("eq=toward")

    return True, " ".join(notes)
