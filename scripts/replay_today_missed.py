"""Replay today's RTH session for missed ORB+HTF signals."""

from __future__ import annotations

from datetime import date, time
from zoneinfo import ZoneInfo

import pandas as pd

from stockpro.config import load_settings
from stockpro.data.bars_5m import filter_premarket, filter_rth, load_cached_bars_5m_ext
from stockpro.spy_day.htf_permission import HtfPermissionConfig, orb_htf_allowed
from stockpro.spy_day.mtf_liquidity import clear_htf_cache, set_htf_cache
from stockpro.spy_day.patterns import best_signal, configure_htf_permission, enrich_bars
from stockpro.spy_day.session import load_spy_day_config

ET = ZoneInfo("America/New_York")


def main() -> None:
    settings = load_settings()
    cfg = load_spy_day_config(settings)
    configure_htf_permission(cfg.htf_permission)

    ext = load_cached_bars_5m_ext(settings)
    rth = filter_rth(ext)
    today = date(2026, 7, 27)
    day = rth[rth.index.date == today]
    if day.empty:
        # try latest session date in cache
        today = rth.index[-1].date()
        day = rth[rth.index.date == today]

    pm = filter_premarket(ext)
    pm_today = pm[pm.index.date == today]

    print(f"=== Session {today} ===")
    print(f"RTH bars so far: {len(day)}  {day.index.min()} -> {day.index.max()}")
    if len(pm_today):
        prior_days = sorted({d for d in rth.index.date if d < today})
        pc = float(rth[rth.index.date == prior_days[-1]].iloc[-1]["Close"]) if prior_days else float("nan")
        print(
            f"Premarket: {len(pm_today)} bars  "
            f"H={pm_today['High'].max():.2f} L={pm_today['Low'].min():.2f} "
            f"last={pm_today.iloc[-1]['Close']:.2f}"
        )
        if pc == pc:
            gap = (float(day.iloc[0]["Open"]) - pc) / pc
            print(f"Gap vs prior close {pc:.2f}: {gap*100:+.2f}%  open={float(day.iloc[0]['Open']):.2f}")
    else:
        print("Premarket: none in cache")

    # Need history before today for OR/ATR/HTF
    hist = rth[rth.index.date <= today]
    enriched = enrich_bars(hist, orb_minutes=cfg.orb_minutes)
    set_htf_cache(enriched)

    day_e = enriched[enriched.index.date == today]
    print(f"\nOR_READY after bar: first True @ ", end="")
    ready = day_e[day_e["OR_READY"] == True]
    if len(ready):
        r0 = ready.iloc[0]
        print(f"{ready.index[0]}  OR_H={r0['OR_HIGH']:.2f} OR_L={r0['OR_LOW']:.2f}")
    else:
        print("not yet")

    # Last ~2 hours window in ET from "now" bar
    last_ts = day_e.index[-1]
    window_start = last_ts - pd.Timedelta(hours=2)
    print(f"\nReplay window: {window_start} -> {last_ts}")

    hits = []
    raw_orb = []
    try:
        for i, ts in enumerate(enriched.index):
            if ts.date() != today:
                continue
            if ts < window_start:
                continue
            row = enriched.iloc[i]
            # Raw ORB without HTF for diagnosis
            configure_htf_permission(HtfPermissionConfig(enabled=False))
            sig_raw = best_signal(
                enriched,
                i,
                enabled=["orb"],
                min_confidence=0.70,
                pattern_min_confidence={"orb": 0.70},
            )
            configure_htf_permission(cfg.htf_permission)
            sig = best_signal(
                enriched,
                i,
                enabled=cfg.patterns,
                min_confidence=cfg.score_threshold,
                pattern_min_confidence=cfg.pattern_min_confidence,
            )
            if sig_raw and sig_raw.pattern == "orb":
                ok, note = orb_htf_allowed(
                    enriched, i, sig_raw.side, float(sig_raw.spot), cfg=cfg.htf_permission
                )
                raw_orb.append(
                    {
                        "ts": str(ts),
                        "side": sig_raw.side,
                        "spot": sig_raw.spot,
                        "reason": sig_raw.reason,
                        "htf_ok": ok,
                        "htf_note": note,
                    }
                )
            if sig is not None:
                hits.append(
                    {
                        "ts": str(ts),
                        "pattern": sig.pattern,
                        "side": sig.side,
                        "conf": sig.confidence,
                        "spot": sig.spot,
                        "reason": sig.reason,
                    }
                )
    finally:
        clear_htf_cache()

    print("\n--- Raw ORB triggers (HTF off) ---")
    if not raw_orb:
        print("None in window")
    for h in raw_orb:
        print(
            f"  {h['ts']}  {h['side'].upper():4s} @ {h['spot']:.2f}  "
            f"htf_ok={h['htf_ok']} ({h['htf_note']})  {h['reason']}"
        )

    print("\n--- Live config signals (ORB + HTF permission) ---")
    if not hits:
        print("None — no trade would have been taken")
    for h in hits:
        print(
            f"  {h['ts']}  {h['pattern']} {h['side'].upper()} "
            f"conf={h['conf']:.2f} @ {h['spot']:.2f}  {h['reason']}"
        )

    # Also full morning (not just 2h) summary
    print("\n--- Full morning price path (first / OR ready / now) ---")
    if len(day_e):
        print(
            f"  09:30 open={float(day_e.iloc[0]['Open']):.2f}  "
            f"now close={float(day_e.iloc[-1]['Close']):.2f}  "
            f"H={float(day_e['High'].max()):.2f} L={float(day_e['Low'].min()):.2f}"
        )
        if len(ready):
            print(
                f"  vs OR: high-break={'YES' if float(day_e['Close'].max()) > float(ready.iloc[0]['OR_HIGH']) else 'no'} "
                f"low-break={'YES' if float(day_e['Close'].min()) < float(ready.iloc[0]['OR_LOW']) else 'no'}"
            )


if __name__ == "__main__":
    main()
