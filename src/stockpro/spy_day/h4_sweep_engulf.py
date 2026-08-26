"""4H liquidity sweep + body engulf (London / NY) — research detector.

Locked (no search):
  - Prior completed 4H candle: High/Low = liquidity, body = max/min(O,C)
  - CALL: sweep below 4H Low, then 5m close above prior body high
  - PUT:  sweep above 4H High, then 5m close below prior body low
  - Windows ET: London 03:00–06:00, NY 09:30–11:30
  - One setup per completed 4H (first confirmation only)
  - pierce = 0 (wick must exceed prior 4H extreme)

Not wired into live patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd

from stockpro.spy_day.mtf_liquidity import resample_ohlcv

ET = ZoneInfo("America/New_York")

LONDON = (time(3, 0), time(6, 0))
NY = (time(9, 30), time(11, 30))


@dataclass
class H4SweepSignal:
    side: str  # call | put
    ts: pd.Timestamp
    spot: float
    reason: str
    h4_end: pd.Timestamp
    session: str  # london | ny


def _in_window(t: time, windows: tuple[tuple[time, time], ...]) -> str | None:
    for name, (a, b) in (("london", LONDON), ("ny", NY)):
        if a <= t <= b:
            return name
    return None


def collect_h4_sweep_engulf(df_5m: pd.DataFrame) -> list[H4SweepSignal]:
    """Scan extended (or RTH) 5m bars for locked 4H sweep+engulf signals."""
    if df_5m.empty:
        return []
    df = df_5m.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize(ET)
    else:
        df.index = df.index.tz_convert(ET)

    h4 = resample_ohlcv(df, "4h")
    if len(h4) < 2:
        return []

    out: list[H4SweepSignal] = []
    # Pair each completed 4H with the next 4H end as search horizon
    ends = list(h4.index)
    for k in range(len(ends) - 1):
        c = h4.iloc[k]
        t0 = ends[k]
        t1 = ends[k + 1]
        h4_hi = float(c["High"])
        h4_lo = float(c["Low"])
        body_hi = max(float(c["Open"]), float(c["Close"]))
        body_lo = min(float(c["Open"]), float(c["Close"]))
        if not (h4_hi > h4_lo > 0 and body_hi >= body_lo):
            continue

        window = df[(df.index > t0) & (df.index <= t1)]
        if window.empty:
            continue

        swept_low = False
        swept_high = False
        fired = False
        for ts, row in window.iterrows():
            t = ts.timetz().replace(tzinfo=None) if hasattr(ts, "timetz") else ts.time()
            sess = _in_window(t, (LONDON, NY))
            if sess is None:
                # still track sweeps outside window so manipulation can occur anytime
                # in the 4H follow-through, confirmation must be in L/NY
                if float(row["Low"]) < h4_lo:
                    swept_low = True
                if float(row["High"]) > h4_hi:
                    swept_high = True
                continue

            if float(row["Low"]) < h4_lo:
                swept_low = True
            if float(row["High"]) > h4_hi:
                swept_high = True

            close = float(row["Close"])
            if swept_low and close > body_hi:
                out.append(
                    H4SweepSignal(
                        side="call",
                        ts=ts,
                        spot=close,
                        reason=f"4H SSL sweep<{h4_lo:.2f} engulf body>{body_hi:.2f}",
                        h4_end=t0,
                        session=sess,
                    )
                )
                fired = True
                break
            if swept_high and close < body_lo:
                out.append(
                    H4SweepSignal(
                        side="put",
                        ts=ts,
                        spot=close,
                        reason=f"4H BSL sweep>{h4_hi:.2f} engulf body<{body_lo:.2f}",
                        h4_end=t0,
                        session=sess,
                    )
                )
                fired = True
                break
        _ = fired
    return out
