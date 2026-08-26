"""Backtest VIX implied-move overlay (change 2). Live paper is NOT changed.

Formula (1-day expected move):
  implied_pct = VIX / 15.87 / 100          # 15.87 ≈ sqrt(252)
  implied_pts = SPY * implied_pct          # SPY points, not raw VIX/15.87*SPX

Close reading: prior VIX close × prior SPY close → next-session expected range.
Open reading:  today's VIX open × today's SPY open → RTH expected range.

Rules tested on top of live 3-lot scale-out, 0DTE only (ORB/retest/power hour):
  hold_k     — do not bank TP until SPY tags open ± k*implied (SL/time still on)
  skip_or    — skip 0DTE if opening-range width > frac * implied
"""

from __future__ import annotations

import json
from datetime import time

import numpy as np
import pandas as pd
import yfinance as yf

from stockpro.config import ROOT, load_settings
from stockpro.data import get_spy_5m
from stockpro.spy_day.backtest import ScaleOutConfig, run_spy_day_backtest
from stockpro.spy_day.session import load_spy_day_config

SQRT252 = 15.87
ET = "America/New_York"


def _yf_daily(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, actions=False)
    if df.empty:
        raise SystemExit(f"No Yahoo data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.title)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def implied_pts(spot: float, vix: float) -> float:
    if spot <= 0 or vix <= 0 or not np.isfinite(vix):
        return float("nan")
    return float(spot) * (float(vix) / SQRT252) / 100.0


def build_daily_implied(spy_5m: pd.DataFrame, vix: pd.DataFrame, spx: pd.DataFrame) -> pd.DataFrame:
    spy = spy_5m.copy()
    if spy.index.tz is None:
        spy.index = spy.index.tz_localize(ET)
    else:
        spy.index = spy.index.tz_convert(ET)
    rows = []
    for d, day in spy.groupby(spy.index.date):
        rth = day.between_time(time(9, 30), time(16, 0))
        if rth.empty:
            continue
        spy_open = float(rth["Open"].iloc[0])
        spy_close = float(rth["Close"].iloc[-1])
        spy_high = float(rth["High"].max())
        spy_low = float(rth["Low"].min())
        n_orb = 6  # 30m
        orb = rth.iloc[:n_orb]
        or_w = float(orb["High"].max() - orb["Low"].min()) if len(orb) else float("nan")
        d_naive = pd.Timestamp(d)
        vix_row = vix.loc[:d_naive].iloc[-1] if len(vix.loc[:d_naive]) else None
        vix_prev = vix.loc[: d_naive - pd.Timedelta(days=1)]
        vix_close_prev = float(vix_prev["Close"].iloc[-1]) if len(vix_prev) else float("nan")
        vix_open = float(vix_row["Open"]) if vix_row is not None else float("nan")
        vix_close = float(vix_row["Close"]) if vix_row is not None else float("nan")
        spx_row = spx.loc[:d_naive].iloc[-1] if len(spx.loc[:d_naive]) else None
        spx_open = float(spx_row["Open"]) if spx_row is not None else float("nan")
        prev_days = spy[spy.index.date < d]
        spy_prior = float(prev_days["Close"].iloc[-1]) if len(prev_days) else float("nan")
        gap = abs(spy_open - spy_prior) if np.isfinite(spy_prior) else float("nan")
        impl_open = implied_pts(spy_open, vix_open)
        impl_close = implied_pts(spy_prior, vix_close_prev)
        realized = max(spy_high - spy_open, spy_open - spy_low)
        rows.append(
            {
                "date": str(d),
                "spy_open": spy_open,
                "spy_close": spy_close,
                "spy_prior_close": spy_prior,
                "spy_high": spy_high,
                "spy_low": spy_low,
                "or_width": or_w,
                "gap": gap,
                "vix_open": vix_open,
                "vix_close": vix_close,
                "vix_prior_close": vix_close_prev,
                "spx_open": spx_open,
                "impl_pts_open": impl_open,
                "impl_pts_close": impl_close,
                "impl_pct_open": (vix_open / SQRT252) if np.isfinite(vix_open) else float("nan"),
                "realized_from_open": realized,
                "realized_vs_impl_open": realized / impl_open if impl_open else float("nan"),
                "or_vs_impl_open": or_w / impl_open if impl_open else float("nan"),
                "gap_vs_impl_close": gap / impl_close if impl_close else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _row(name: str, result) -> dict:
    m = result.metrics
    trades = result.trades
    return {
        "variant": name,
        "n_trades": int(m.get("n_trades", 0)),
        "win_rate": float(m.get("win_rate", 0)),
        "expectancy": float(m.get("expectancy", 0)),
        "profit_factor": float(m.get("profit_factor", 0)),
        "total_pnl": float(trades["pnl"].sum()) if len(trades) else 0.0,
        "max_drawdown": float(m.get("max_drawdown", 0)),
        "orb_n": int((trades["pattern"] == "orb").sum()) if len(trades) else 0,
        "orb_pnl": float(trades.loc[trades["pattern"] == "orb", "pnl"].sum()) if len(trades) else 0.0,
    }


def _print_by_pattern(label: str, result) -> None:
    print(f"{label} by pattern:")
    for pat, stats in sorted(result.by_pattern.items()):
        print(
            f"  {pat}: n={int(stats['n_trades'])} WR={stats['win_rate']:.1%} "
            f"E=${stats['expectancy']:.2f} PF={stats['profit_factor']:.2f} "
            f"PnL=${stats['total_pnl']:.0f}"
        )


def make_unlock_fn(daily: pd.DataFrame, closes: pd.Series, k: float, ref: str, zerodte_only: bool = True):
    by_date = {r["date"]: r for r in daily.to_dict(orient="records")}

    def _fn(ts, side, is_amd, df, i):
        if zerodte_only and is_amd:
            return None
        rec = by_date.get(str(ts.date()))
        if not rec:
            return None
        impl = rec["impl_pts_open"] if ref == "open" else rec["impl_pts_close"]
        if not impl or not np.isfinite(impl):
            return None
        ref_px = rec["spy_open"] if ref == "open" else rec["spy_prior_close"]
        if not np.isfinite(ref_px):
            return None
        target = ref_px + side * k * impl
        loc = closes.index.get_loc(ts)
        if isinstance(loc, slice):
            return None
        # Already delivered at/before entry → take TP as usual
        if side == 1 and float(closes.iloc[int(loc)]) >= target:
            return None
        if side == -1 and float(closes.iloc[int(loc)]) <= target:
            return None
        day = ts.date()
        for j in range(int(loc) + 1, len(closes)):
            if closes.index[j].date() != day:
                return len(closes)  # never unlock same day
            px = float(closes.iloc[j])
            if side == 1 and px >= target:
                return j
            if side == -1 and px <= target:
                return j
        return len(closes)

    return _fn


def make_skip_or(daily: pd.DataFrame, frac: float):
    by_date = {r["date"]: r for r in daily.to_dict(orient="records")}

    def _fn(sig, i, df):
        if sig.pattern not in {"orb", "orb_retest"}:
            return False
        rec = by_date.get(str(df.index[i].date()))
        if not rec or not rec.get("impl_pts_open"):
            return False
        or_w = rec["or_width"]
        return bool(or_w / rec["impl_pts_open"] > frac)

    return _fn


def aug4_case(daily: pd.DataFrame, spy: pd.DataFrame, trades: pd.DataFrame) -> None:
    rec = daily[daily["date"] == "2026-08-04"]
    print("=== Aug 4 2026 implied move ===")
    if rec.empty:
        print("no daily row")
        return
    r = rec.iloc[0]
    print(
        f"VIX open {r['vix_open']:.2f}  prior close {r['vix_prior_close']:.2f}  "
        f"SPY open {r['spy_open']:.2f}  prior close {r['spy_prior_close']:.2f}"
    )
    print(
        f"Implied SPY pts: open-reading {r['impl_pts_open']:.2f}  "
        f"close-reading {r['impl_pts_close']:.2f}  ({r['impl_pct_open']:.2f}% of SPY)"
    )
    print(
        f"OR width {r['or_width']:.2f} ({100*r['or_vs_impl_open']:.0f}% of implied)  "
        f"gap {r['gap']:.2f}  realized from open {r['realized_from_open']:.2f} "
        f"({r['realized_vs_impl_open']:.2f}x implied)"
    )
    print(f"Day range {r['spy_low']:.2f}–{r['spy_high']:.2f}  close {r['spy_close']:.2f}")
    if trades is None or trades.empty:
        return
    t = trades.copy()
    t["date"] = t["date"].astype(str)
    day_tr = t[t["date"] == "2026-08-04"]
    print("Baseline trades that day:")
    cols = [c for c in ["datetime", "pattern", "side", "spot", "pnl", "exit_reason"] if c in day_tr.columns]
    if len(day_tr):
        print(day_tr[cols].to_string(index=False))
    else:
        print("  (none in this variant)")


def main() -> None:
    settings = load_settings()
    cfg = load_spy_day_config(settings)
    bars = get_spy_5m(settings, refresh=False, rth_only=True)
    if bars.empty:
        raise SystemExit("No 5m bars")
    start = str(pd.Timestamp(bars.index.min()).date())
    end = str(pd.Timestamp(bars.index.max()).date() + pd.Timedelta(days=2))
    print("Downloading ^VIX and ^GSPC", start, "->", end)
    vix = _yf_daily("^VIX", start, end)
    spx = _yf_daily("^GSPC", start, end)
    daily = build_daily_implied(bars, vix, spx)
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_dir / "implied_move_daily.csv", index=False)

    scale = ScaleOutConfig(qty=3, tp2_pct=0.60, runner_stop_at_entry=True)
    closes = bars["Close"]
    if closes.index.tz is None:
        closes.index = closes.index.tz_localize(ET)

    variants = {}
    print("Running baseline 3c_scale...")
    variants["3c_scale"] = run_spy_day_backtest(bars, cfg=cfg, scale_out=scale)

    print("Running hold TP until 0.5x implied (open)...")
    variants["hold_0.5x_open"] = run_spy_day_backtest(
        bars,
        cfg=cfg,
        scale_out=scale,
        tp_unlock_fn=make_unlock_fn(daily, closes, 0.5, "open"),
    )
    print("Running hold TP until 1.0x implied (open)...")
    variants["hold_1.0x_open"] = run_spy_day_backtest(
        bars,
        cfg=cfg,
        scale_out=scale,
        tp_unlock_fn=make_unlock_fn(daily, closes, 1.0, "open"),
    )
    print("Running hold TP until 0.5x implied (prior close)...")
    variants["hold_0.5x_close"] = run_spy_day_backtest(
        bars,
        cfg=cfg,
        scale_out=scale,
        tp_unlock_fn=make_unlock_fn(daily, closes, 0.5, "prior_close"),
    )
    print("Running skip ORB if OR > 50% implied...")
    variants["skip_or_50"] = run_spy_day_backtest(
        bars,
        cfg=cfg,
        scale_out=scale,
        skip_fn=make_skip_or(daily, 0.50),
    )
    print("Running skip OR>50% + hold 0.5x...")
    variants["skip_or50_hold_0.5"] = run_spy_day_backtest(
        bars,
        cfg=cfg,
        scale_out=scale,
        skip_fn=make_skip_or(daily, 0.50),
        tp_unlock_fn=make_unlock_fn(daily, closes, 0.5, "open"),
    )

    summary = pd.DataFrame([_row(k, v) for k, v in variants.items()])
    summary.to_csv(out_dir / "implied_move_research.csv", index=False)
    variants["hold_0.5x_open"].trades.to_csv(out_dir / "implied_move_hold05_trades.csv", index=False)

    print()
    print("Bars:", len(bars), bars.index.min(), "->", bars.index.max())
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    aug4_case(daily, bars, variants["3c_scale"].trades)
    print()
    print("--- Aug 4 under hold_0.5x_open ---")
    aug4_case(daily, bars, variants["hold_0.5x_open"].trades)
    print()
    _print_by_pattern("3c_scale", variants["3c_scale"])
    print()
    _print_by_pattern("hold_0.5x_open", variants["hold_0.5x_open"])
    print()
    _print_by_pattern("skip_or_50", variants["skip_or_50"])

    payload = {
        "summary": summary.to_dict(orient="records"),
        "formula": "implied_pts = SPY * (VIX/15.87)/100",
        "live_unchanged": True,
        "by_pattern_baseline": variants["3c_scale"].by_pattern,
        "by_pattern_hold05": variants["hold_0.5x_open"].by_pattern,
    }

    def _san(o):
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items()}
        if isinstance(o, float) and (o == float("inf") or not np.isfinite(o)):
            return None
        return o

    (out_dir / "implied_move_research.json").write_text(
        json.dumps(_san(payload), indent=2), encoding="utf-8"
    )
    print()
    print("Wrote", out_dir / "implied_move_research.csv")
    print("Wrote", out_dir / "implied_move_daily.csv")


if __name__ == "__main__":
    main()
