"""PDH/PDL liquidity concurrence filter (locked).

ICT draw-on-liquidity idea: prefer sells after buyside (PDH) is tagged,
buys after sellside (PDL) is tagged — earlier in the same session.

Locked:
  pierce_atr = 0.15
  apply_to = orb, orb_retest, power_hour, amd
  CALL requires session Low <= PDL + pierce*ATR (tagged/swept PDL)
  PUT  requires session High >= PDH - pierce*ATR (tagged/swept PDH)
  if PDH/PDL missing → allow

Default enabled=False (research only).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stockpro.spy_day.patterns import PatternSignal

PDH_PARAMS: dict[str, Any] = {
    "enabled": False,
    "pierce_atr": 0.15,
    "apply_to": ("orb", "orb_retest", "power_hour", "amd"),
}


def configure_pdh_params(raw: dict | None) -> None:
    if not raw:
        return
    for k, v in raw.items():
        if k not in PDH_PARAMS:
            continue
        if k == "apply_to":
            PDH_PARAMS[k] = tuple(v)
        else:
            PDH_PARAMS[k] = v


def pdh_allows(df: pd.DataFrame, i: int, sig: PatternSignal) -> tuple[bool, str]:
    if not PDH_PARAMS.get("enabled"):
        return True, ""
    if sig.pattern not in set(PDH_PARAMS.get("apply_to") or ()):
        return True, ""

    row = df.iloc[i]
    pdh, pdl = row.get("PDH"), row.get("PDL")
    if pdh is None or pdl is None or pd.isna(pdh) or pd.isna(pdl):
        return True, "pdh_missing"
    pdh_f, pdl_f = float(pdh), float(pdl)
    atr_raw = row.get("ATR")
    atr = float(atr_raw) if atr_raw is not None and pd.notna(atr_raw) else np.nan
    if pd.isna(atr) or atr <= 0:
        return True, "pdh_no_atr"
    pierce = float(PDH_PARAMS.get("pierce_atr", 0.15)) * atr

    day = row.get("session_date")
    same = df[(df["session_date"] == day) & (df.index <= df.index[i])]
    if same.empty:
        return True, "pdh_empty"
    sess_hi = float(same["High"].max())
    sess_lo = float(same["Low"].min())

    if sig.side == "call":
        if sess_lo <= pdl_f + pierce:
            return True, f"pdh_ssl_tag PDL={pdl_f:.2f} lo={sess_lo:.2f}"
        return False, f"pdh_block_call need_PDL_tag PDL={pdl_f:.2f} lo={sess_lo:.2f}"
    if sig.side == "put":
        if sess_hi >= pdh_f - pierce:
            return True, f"pdh_bsl_tag PDH={pdh_f:.2f} hi={sess_hi:.2f}"
        return False, f"pdh_block_put need_PDH_tag PDH={pdh_f:.2f} hi={sess_hi:.2f}"
    return True, ""
