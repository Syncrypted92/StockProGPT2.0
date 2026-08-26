"""SMT (Smart Money Technique) — locked SPY vs QQQ divergence filter.

At signal time, from session open:
  CALL: QQQ took a deeper relative low than SPY (SSL on Nasdaq, SPY held)
  PUT:  QQQ took a higher relative high than SPY (BSL on Nasdaq, SPY held)

Locked:
  min_div = 0.0008 (~8 bps of open)
  apply_to = orb, orb_retest, power_hour, amd
  if QQQ bars missing → allow

Default enabled=False (research only).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stockpro.config import ROOT
from stockpro.spy_day.patterns import PatternSignal

SMT_PARAMS: dict[str, Any] = {
    "enabled": False,
    "min_div": 0.0008,
    "apply_to": ("orb", "orb_retest", "power_hour", "amd"),
}

_QQQ: pd.DataFrame | None = None
_EXTREMES: dict | None = None


def configure_smt_params(raw: dict | None) -> None:
    global _EXTREMES
    if not raw:
        return
    for k, v in raw.items():
        if k not in SMT_PARAMS:
            continue
        if k == "apply_to":
            SMT_PARAMS[k] = tuple(v)
        else:
            SMT_PARAMS[k] = v
    _EXTREMES = None


def clear_smt_cache() -> None:
    global _QQQ, _EXTREMES
    _QQQ = None
    _EXTREMES = None


def _load_qqq() -> pd.DataFrame | None:
    global _QQQ
    if _QQQ is not None:
        return _QQQ
    path = ROOT / "data" / "bars" / "QQQ_5m.parquet"
    if not path.exists():
        return None
    q = pd.read_parquet(path)
    if q.index.tz is None:
        q.index = q.index.tz_localize("America/New_York")
    _QQQ = q
    return _QQQ


def _qqq_extremes(spy_index: pd.DatetimeIndex) -> dict | None:
    global _EXTREMES
    if _EXTREMES is not None:
        return _EXTREMES
    q = _load_qqq()
    if q is None:
        return None
    common = spy_index.intersection(q.index)
    if len(common) < 50:
        return None
    qq = q.loc[common].copy()
    day = pd.Series(qq.index.date, index=qq.index)
    sess_open = qq["Open"].groupby(day).transform("first")
    run_hi = qq["High"].groupby(day).cummax()
    run_lo = qq["Low"].groupby(day).cummin()
    up = (run_hi - sess_open) / sess_open.replace(0, np.nan)
    dn = (sess_open - run_lo) / sess_open.replace(0, np.nan)
    _EXTREMES = {
        ts: (float(u), float(d))
        for ts, u, d in zip(up.index, up.values, dn.values)
        if pd.notna(u) and pd.notna(d)
    }
    return _EXTREMES


def _spy_extremes(df: pd.DataFrame, i: int) -> tuple[float, float] | None:
    row = df.iloc[i]
    day = row.get("session_date")
    if day is None:
        return None
    same = df[(df["session_date"] == day) & (df.index <= df.index[i])]
    if same.empty:
        return None
    o = float(same.iloc[0]["Open"])
    if o <= 0:
        return None
    hi = float(same["High"].max())
    lo = float(same["Low"].min())
    return (hi - o) / o, (o - lo) / o


def smt_allows(df: pd.DataFrame, i: int, sig: PatternSignal) -> tuple[bool, str]:
    if not SMT_PARAMS.get("enabled"):
        return True, ""
    if sig.pattern not in set(SMT_PARAMS.get("apply_to") or ()):
        return True, ""
    qx = _qqq_extremes(df.index)
    if qx is None:
        return True, "smt_no_qqq"
    ts = df.index[i]
    if ts not in qx:
        day = df.iloc[i].get("session_date")
        cands = [t for t in qx if getattr(t, "date", lambda: None)() == day and t <= ts]
        if not cands:
            return True, "smt_no_ts"
        ts = max(cands)
    spy = _spy_extremes(df, i)
    if spy is None:
        return True, "smt_no_spy"
    spy_up, spy_dn = spy
    q_up, q_dn = qx[ts]
    div = float(SMT_PARAMS.get("min_div", 0.0008))

    if sig.side == "call":
        if q_dn >= spy_dn + div:
            return True, f"smt_bull q_dn={q_dn*100:.2f}% spy_dn={spy_dn*100:.2f}%"
        return False, f"smt_block_call q_dn={q_dn*100:.2f}% spy_dn={spy_dn*100:.2f}%"
    if sig.side == "put":
        if q_up >= spy_up + div:
            return True, f"smt_bear q_up={q_up*100:.2f}% spy_up={spy_up*100:.2f}%"
        return False, f"smt_block_put q_up={q_up*100:.2f}% spy_up={spy_up*100:.2f}%"
    return True, ""
