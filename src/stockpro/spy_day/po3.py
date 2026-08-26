"""Power of Three (PO3) — locked Judas/open-manipulation bias.

ICT idea: morning accumulation → one-sided manipulation (false run) → distribution
the other way. We use only the pre-OR window (first 30m / 9:30–9:55) vs session open.

Locked (no search):
  pierce_atr = 0.15  (same scale as AMD)
  apply_to = orb, orb_retest, power_hour  (AMD already is PO3/AMD — leave alone)
  neutral_policy = allow  (only filter when Judas is one-sided)

Research-only until walk-forward clears it — default enabled=False.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stockpro.spy_day.patterns import PatternSignal

PO3_PARAMS: dict[str, Any] = {
    "enabled": False,
    "pierce_atr": 0.15,
    "apply_to": ("orb", "orb_retest", "power_hour"),
    "neutral_policy": "allow",  # allow | block
    "accum_bars": 6,  # 30m @ 5m
}


def configure_po3_params(raw: dict | None) -> None:
    if not raw:
        return
    for k, v in raw.items():
        if k not in PO3_PARAMS:
            continue
        if k == "apply_to":
            PO3_PARAMS[k] = tuple(v)
        else:
            PO3_PARAMS[k] = v


def session_po3_bias(df: pd.DataFrame, i: int) -> tuple[str | None, str]:
    """Preferred distribution side after morning Judas: 'call' | 'put' | None."""
    if i < 0 or i >= len(df):
        return None, "po3_bad_i"
    row = df.iloc[i]
    day = row.get("session_date")
    if day is None:
        return None, "po3_no_day"
    day_all = df[df["session_date"] == day]
    if day_all.empty:
        return None, "po3_empty_day"
    n = int(PO3_PARAMS.get("accum_bars", 6))
    accum = day_all.iloc[:n]
    if len(accum) < max(3, n // 2):
        return None, "po3_insufficient"

    sess_open = float(accum.iloc[0]["Open"])
    hi = float(accum["High"].max())
    lo = float(accum["Low"].min())
    atr_raw = accum.iloc[-1].get("ATR")
    atr = float(atr_raw) if atr_raw is not None and pd.notna(atr_raw) else np.nan
    if pd.isna(atr) or atr <= 0:
        return None, "po3_no_atr"

    pierce = float(PO3_PARAMS["pierce_atr"]) * atr
    up = hi >= sess_open + pierce
    dn = lo <= sess_open - pierce

    if up and not dn:
        # Buys induced above open → distribute down
        return "put", f"po3_judas_up open={sess_open:.2f} hi={hi:.2f}"
    if dn and not up:
        # Sells induced below open → distribute up
        return "call", f"po3_judas_dn open={sess_open:.2f} lo={lo:.2f}"
    if up and dn:
        return None, "po3_both_sweeps"
    return None, "po3_neutral"


def po3_allows(df: pd.DataFrame, i: int, sig: PatternSignal) -> tuple[bool, str]:
    if not PO3_PARAMS.get("enabled"):
        return True, ""
    apply_to = set(PO3_PARAMS.get("apply_to") or ())
    if sig.pattern not in apply_to:
        return True, ""
    bias, note = session_po3_bias(df, i)
    if bias is None:
        if PO3_PARAMS.get("neutral_policy", "allow") == "block":
            return False, note or "po3_neutral_block"
        return True, note
    if sig.side != bias:
        return False, f"po3_block want={bias} got={sig.side} ({note})"
    return True, note
