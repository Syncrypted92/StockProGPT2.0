"""AMD on NASDAQ futures (NQ / MNQ) vs SPY — same session logic, points P&L.

Yahoo 5m futures history is short (~60d). MNQ and NQ prices track the same index;
only $/point differs (MNQ=$2, NQ=$20).
"""

from __future__ import annotations

import json
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from stockpro.spy_day.amd import AMD_PARAMS, amd_signal, configure_amd_params
from stockpro.spy_day.htf_permission import HtfPermissionConfig
from stockpro.spy_day.mtf_liquidity import clear_htf_cache, set_htf_cache
from stockpro.spy_day.patterns import configure_htf_permission, enrich_bars

ROOT = Path(__file__).resolve().parents[1]
BARS = ROOT / "data" / "bars"


def _download_5m(symbol: str) -> pd.DataFrame:
    raw = yf.download(symbol, period="60d", interval="5m", progress=False, auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")
    return df.sort_index()


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    t = df.index.time
    mask = (t >= time(9, 30)) & (t <= time(15, 55))
    # weekdays only
    out = df[mask]
    out = out[out.index.dayofweek < 5]
    return out


def _pf(s: pd.Series) -> float:
    w = float(s[s > 0].sum())
    l = float((-s[s < 0]).sum())
    if l <= 1e-12:
        return float("inf") if w > 0 else 0.0
    return w / l


def _stats(pnls: list[float]) -> dict:
    if not pnls:
        return {"n": 0, "wr": 0.0, "exp": 0.0, "pf": 0.0, "pnl": 0.0}
    s = pd.Series(pnls, dtype=float)
    return {"n": int(len(s)), "wr": float((s > 0).mean()), "exp": float(s.mean()), "pf": _pf(s), "pnl": float(s.sum())}


def collect_amd(df: pd.DataFrame, *, confirm: bool, htf: bool, skip_friday: bool) -> list[dict]:
    configure_amd_params(
        {
            "window_start": time(10, 0),
            "window_end": time(12, 0),
            "pierce_atr": 0.15,
            "one_per_day": True,
            "require_confirm": confirm,
            "apply_htf": htf,
            "min_body_frac": 0.35,
        }
    )
    configure_htf_permission(
        HtfPermissionConfig(enabled=True, skip_4h_counter_trend=True, eq_context="none")
        if htf
        else HtfPermissionConfig(enabled=False)
    )
    out: list[dict] = []
    for i in range(len(df)):
        sig = amd_signal(df, i)
        if not sig:
            continue
        ts = df.index[i]
        if skip_friday and ts.weekday() == 4:
            continue
        if out and out[-1]["ts"].date() == ts.date():
            continue
        atr = float(df.iloc[i]["ATR"]) if pd.notna(df.iloc[i].get("ATR")) else np.nan
        out.append(
            {
                "i": i,
                "ts": ts,
                "side": sig.side,
                "spot": float(sig.spot),
                "atr": atr,
            }
        )
    return out


def sim_points(df: pd.DataFrame, sig: dict, point_value: float) -> float:
    """1.5R / 1R ATR, same-day force flat @ 15:45."""
    atr = sig["atr"]
    if atr != atr or atr <= 0:
        return 0.0
    side = 1 if sig["side"] == "call" else -1
    entry = float(sig["spot"])
    stop = entry - side * 1.0 * atr
    target = entry + side * 1.5 * atr
    exit_px = entry
    for j in range(sig["i"] + 1, len(df)):
        ts = df.index[j]
        if ts.date() != sig["ts"].date():
            exit_px = float(df.iloc[j - 1]["Close"])
            break
        hi, lo, c = float(df.iloc[j]["High"]), float(df.iloc[j]["Low"]), float(df.iloc[j]["Close"])
        if side > 0:
            hit_stop, hit_tgt = lo <= stop, hi >= target
        else:
            hit_stop, hit_tgt = hi >= stop, lo <= target
        if hit_stop:
            exit_px = stop
            break
        if hit_tgt:
            exit_px = target
            break
        if ts.time() >= time(15, 45):
            exit_px = c
            break
        exit_px = c
    return float((exit_px - entry) * side * point_value)


def run_symbol(name: str, df_rth: pd.DataFrame, point_value: float) -> list[dict]:
    print(f"\n=== {name} RTH bars={len(df_rth)} {df_rth.index.min()} -> {df_rth.index.max()} pts=${point_value} ===")
    enriched = enrich_bars(df_rth, orb_minutes=30)
    set_htf_cache(enriched)
    rows = []
    try:
        for label, confirm, htf, skip_fri in [
            ("reclaim", False, False, False),
            ("reclaim_htf", False, True, False),
            ("reclaim_htf_noFri", False, True, True),
            ("confirm_htf", True, True, False),
            ("confirm_htf_noFri", True, True, True),
        ]:
            sigs = collect_amd(enriched, confirm=confirm, htf=htf, skip_friday=skip_fri)
            pnls = [sim_points(enriched, s, point_value) for s in sigs]
            st = _stats(pnls)
            row = {"symbol": name, "variant": label, "point_value": point_value, **st}
            rows.append(row)
            print(
                f"  {label:<20} n={st['n']:<3} WR={st['wr']*100:5.1f}% "
                f"PF={st['pf']:.2f} exp=${st['exp']:.2f} pnl=${st['pnl']:.0f}"
            )
    finally:
        clear_htf_cache()
        configure_amd_params({"require_confirm": False, "apply_htf": False})
    return rows


def main() -> None:
    BARS.mkdir(parents=True, exist_ok=True)
    print("Downloading Yahoo 5m futures (max ~60d)...")
    nq = _download_5m("NQ=F")
    mnq = _download_5m("MNQ=F")
    # Also SPY over overlap for fair compare
    spy = pd.read_parquet(BARS / "SPY_5m.parquet")
    if spy.index.tz is None:
        spy.index = spy.index.tz_localize("America/New_York")

    nq_rth = filter_rth(nq)
    mnq_rth = filter_rth(mnq)
    nq_rth.to_parquet(BARS / "NQ_5m_rth.parquet")
    mnq_rth.to_parquet(BARS / "MNQ_5m_rth.parquet")
    print(f"NQ RTH={len(nq_rth)}  MNQ RTH={len(mnq_rth)}")

    start = max(nq_rth.index.min(), mnq_rth.index.min())
    end = min(nq_rth.index.max(), mnq_rth.index.max())
    spy_rth = spy[(spy.index >= start) & (spy.index <= end)]
    print(f"Overlap window {start} -> {end}")
    print(f"SPY overlap bars={len(spy_rth)}")

    all_rows: list[dict] = []
    # MNQ $2/pt, NQ $20/pt — prices nearly identical so signal path matches
    all_rows += run_symbol("MNQ", mnq_rth, point_value=2.0)
    all_rows += run_symbol("NQ", nq_rth, point_value=20.0)
    # SPY $5/pt MES-style proxy on same date window
    all_rows += run_symbol("SPY_proxy_MES", spy_rth, point_value=5.0)

    print("\n=== Ranked by PF (n>=15) ===")
    ranked = sorted(
        [r for r in all_rows if r["n"] >= 15],
        key=lambda r: (r["pf"] if r["pf"] == r["pf"] else 0, r["exp"]),
        reverse=True,
    )
    for r in ranked:
        print(
            f"{r['symbol']:<16} {r['variant']:<20} n={r['n']:<3} "
            f"WR={r['wr']*100:5.1f}% PF={r['pf']:.2f} pnl=${r['pnl']:.0f}"
        )

    out = ROOT / "artifacts" / "amd_nq_mnq_research.csv"
    pd.DataFrame(all_rows).to_csv(out, index=False)
    (ROOT / "artifacts" / "amd_nq_mnq_research.json").write_text(
        json.dumps(
            {
                "note": "Yahoo 5m ~60d only; MNQ/NQ same index path, different $/pt",
                "window": {"start": str(start), "end": str(end)},
                "rows": all_rows,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
