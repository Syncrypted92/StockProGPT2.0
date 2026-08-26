"""Backtest morning 15m high/low vs live 30m ORB. Live paper is NOT changed.

15m open range = first three 5m bars (09:30–09:45 ET).
Live ORB = first 30m (09:30–10:00).

On top of 3-lot scale-out + live patterns:
  15m_or        — replace 30m box with 15m (ORB, retest, AMD all use 15m)
  15m_bias      — keep 30m ORB; only take ORB/retest with the 15m candle's direction
  15m_broke     — keep 30m ORB; only take if 15m H/L was already tagged after 09:45
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import time

import pandas as pd

from stockpro.config import ROOT, load_settings
from stockpro.data import get_spy_5m
from stockpro.spy_day.backtest import ScaleOutConfig, run_spy_day_backtest
from stockpro.spy_day.session import load_spy_day_config

ET = "America/New_York"


def _row(name: str, result) -> dict:
    m = result.metrics
    trades = result.trades
    orb = trades[trades["pattern"] == "orb"] if len(trades) else trades
    return {
        "variant": name,
        "n_trades": int(m.get("n_trades", 0)),
        "win_rate": float(m.get("win_rate", 0)),
        "expectancy": float(m.get("expectancy", 0)),
        "profit_factor": float(m.get("profit_factor", 0)),
        "total_pnl": float(trades["pnl"].sum()) if len(trades) else 0.0,
        "max_drawdown": float(m.get("max_drawdown", 0)),
        "orb_n": int(len(orb)),
        "orb_wr": float((orb["pnl"] > 0).mean()) if len(orb) else 0.0,
        "orb_pnl": float(orb["pnl"].sum()) if len(orb) else 0.0,
    }


def _print_by_pattern(label: str, result) -> None:
    print(f"{label} by pattern:")
    for pat, stats in sorted(result.by_pattern.items()):
        print(
            f"  {pat}: n={int(stats['n_trades'])} WR={stats['win_rate']:.1%} "
            f"E=${stats['expectancy']:.2f} PF={stats['profit_factor']:.2f} "
            f"PnL=${stats['total_pnl']:.0f}"
        )


def _open15(df: pd.DataFrame, i: int) -> dict | None:
    """First 15m of that RTH session: bars 0,1,2."""
    day = df.index[i].date()
    same = df[df.index.date == day]
    if len(same) < 3:
        return None
    w = same.iloc[:3]
    o = float(w["Open"].iloc[0])
    h = float(w["High"].max())
    l = float(w["Low"].min())
    c = float(w["Close"].iloc[-1])
    return {"open": o, "high": h, "low": l, "close": c, "bull": c > o, "bear": c < o}


def skip_15m_bias(sig, i, df) -> bool:
    if sig.pattern not in {"orb", "orb_retest"}:
        return False
    s = _open15(df, i)
    if s is None:
        return False
    if sig.side == "call" and not s["bull"]:
        return True
    if sig.side == "put" and not s["bear"]:
        return True
    return False


def skip_15m_not_broken(sig, i, df) -> bool:
    """Skip ORB unless 15m high (call) or low (put) was tagged after 09:45."""
    if sig.pattern not in {"orb", "orb_retest"}:
        return False
    s = _open15(df, i)
    if s is None:
        return False
    day = df.index[i].date()
    same = df[df.index.date == day]
    try:
        pos = list(same.index).index(df.index[i])
    except ValueError:
        return False
    after = same.iloc[3 : pos + 1]
    if after.empty:
        return True
    if sig.side == "call":
        return not bool((after["High"] >= s["high"]).any())
    return not bool((after["Low"] <= s["low"]).any())


def aug4_15m(bars: pd.DataFrame) -> None:
    d = bars.loc["2026-08-04"]
    if d.empty:
        print("No Aug 4 bars")
        return
    w = d.iloc[:3]
    o, h, l, c = float(w["Open"].iloc[0]), float(w["High"].max()), float(w["Low"].min()), float(w["Close"].iloc[-1])
    w30 = d.iloc[:6]
    print("=== Aug 4 15m open (09:30–09:45) ===")
    print(f"O {o:.2f}  H {h:.2f}  L {l:.2f}  C {c:.2f}  {'bull' if c>o else 'bear' if c<o else 'doji'}")
    print(
        f"30m OR  H {float(w30['High'].max()):.2f}  L {float(w30['Low'].min()):.2f}  "
        f"(live ORB broke 763.47 at 10:05)"
    )


def main() -> None:
    settings = load_settings()
    cfg = load_spy_day_config(settings)
    bars = get_spy_5m(settings, refresh=False, rth_only=True)
    if bars.empty:
        raise SystemExit("No 5m bars")
    scale = ScaleOutConfig(qty=3, tp2_pct=0.60, runner_stop_at_entry=True)
    cfg15 = replace(cfg, orb_minutes=15)

    print("Bars:", len(bars), bars.index.min(), "->", bars.index.max())
    aug4_15m(bars)

    variants = {}
    print("baseline 30m 3c_scale...")
    variants["30m_live"] = run_spy_day_backtest(bars, cfg=cfg, scale_out=scale)
    print("15m OR replace...")
    variants["15m_or"] = run_spy_day_backtest(bars, cfg=cfg15, scale_out=scale)
    print("30m + 15m candle bias...")
    variants["15m_bias"] = run_spy_day_backtest(
        bars, cfg=cfg, scale_out=scale, skip_fn=skip_15m_bias
    )
    print("30m + 15m H/L already tagged...")
    variants["15m_broke"] = run_spy_day_backtest(
        bars, cfg=cfg, scale_out=scale, skip_fn=skip_15m_not_broken
    )

    summary = pd.DataFrame([_row(k, v) for k, v in variants.items()])
    out = ROOT / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "open15m_research.csv", index=False)
    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    for name in variants:
        _print_by_pattern(name, variants[name])
        print()

    payload = {
        "summary": summary.to_dict(orient="records"),
        "live_unchanged": True,
        "note": "15m_or also shrinks the AMD box to 15m because AMD uses the same OR.",
    }

    def _san(o):
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items()}
        if isinstance(o, float) and o == float("inf"):
            return None
        return o

    (out / "open15m_research.json").write_text(json.dumps(_san(payload), indent=2), encoding="utf-8")
    print("Wrote", out / "open15m_research.csv")


if __name__ == "__main__":
    main()
