"""Research overnight (prior-close → RTH open) gap fills for SPY 0DTE."""

from __future__ import annotations

import json
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from stockpro.spy_day.backtest import _metrics, _simulate_0dte_path
from stockpro.spy_day.patterns import enrich_bars
from stockpro.spy_day.session import load_spy_day_config

ROOT = Path(__file__).resolve().parents[1]


def gap_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per session: gap vs prior RTH close, whether prior close was tagged."""
    rows = []
    days = sorted({ts.date() for ts in df.index})
    for i, d in enumerate(days):
        if i == 0:
            continue
        day = df[df.index.date == d]
        prev = df[df.index.date == days[i - 1]]
        if len(day) < 10 or len(prev) < 5:
            continue
        o = float(day.iloc[0]["Open"])
        pc = float(prev.iloc[-1]["Close"])
        gap = (o - pc) / pc
        hi = float(day["High"].max())
        lo = float(day["Low"].min())
        filled = lo <= pc <= hi
        touch = None
        for ts, r in day.iterrows():
            if float(r["Low"]) <= pc <= float(r["High"]):
                touch = ts
                break
        # first 30m direction toward fill
        early = day.iloc[:6]
        early_toward = False
        if gap > 0:
            early_toward = float(early["Low"].min()) < o
        elif gap < 0:
            early_toward = float(early["High"].max()) > o
        rows.append(
            {
                "date": str(d),
                "gap_pct": gap,
                "gap_abs": abs(gap),
                "filled": filled,
                "touch": str(touch) if touch is not None else None,
                "touch_time": touch.strftime("%H:%M") if touch is not None else None,
                "open": o,
                "prior_close": pc,
                "day_close": float(day.iloc[-1]["Close"]),
                "early_toward_fill": early_toward,
            }
        )
    return pd.DataFrame(rows)


def run_gap_fill_backtest(
    df: pd.DataFrame,
    *,
    min_gap_pct: float,
    max_gap_pct: float,
    entry_mode: str,
    entry_latest: time,
    require_early_toward: bool,
    htf_skip_counter: bool,
) -> dict:
    """
    entry_mode:
      open     — enter on first bar (09:30) fade toward prior close
      confirm  — enter when price makes a 5m close toward the gap (first such bar)
      orb_align — only if ORB break is in gap-fill direction
    """
    from stockpro.spy_day.mtf_liquidity import build_mtf_map, clear_htf_cache, set_htf_cache

    cfg = load_spy_day_config()
    force_flat = time(*map(int, cfg.force_flat_et.split(":")[:2]))
    set_htf_cache(df)
    equity = 100_000.0
    trades: list[dict] = []
    equity_rows: list[dict] = []
    closes = df["Close"]

    days = sorted({ts.date() for ts in df.index})
    try:
        for i, d in enumerate(days):
            if i == 0:
                continue
            day = df[df.index.date == d]
            prev = df[df.index.date == days[i - 1]]
            if len(day) < 15:
                continue
            o = float(day.iloc[0]["Open"])
            pc = float(prev.iloc[-1]["Close"])
            gap = (o - pc) / pc
            gap_abs = abs(gap)
            if gap_abs < min_gap_pct or gap_abs > max_gap_pct:
                continue

            # Fade the gap: gap up → put (fill down); gap down → call (fill up)
            side_str = "put" if gap > 0 else "call"
            side = -1 if side_str == "put" else 1

            if require_early_toward:
                early = day.iloc[:6]
                if gap > 0 and float(early["Low"].min()) >= o:
                    continue
                if gap < 0 and float(early["High"].max()) <= o:
                    continue

            entry_ts = None
            if entry_mode == "open":
                entry_ts = day.index[0]
            elif entry_mode == "confirm":
                for ts, r in day.iterrows():
                    if ts.time() > entry_latest:
                        break
                    if ts.time() < time(9, 35):
                        continue
                    c = float(r["Close"])
                    if gap > 0 and c < o:  # moving down toward fill
                        entry_ts = ts
                        break
                    if gap < 0 and c > o:
                        entry_ts = ts
                        break
            elif entry_mode == "orb_align":
                # wait until OR ready; take first ORB break in fill direction only
                for j, (ts, r) in enumerate(day.iterrows()):
                    if not bool(r.get("OR_READY", False)):
                        continue
                    if ts.time() > entry_latest:
                        break
                    oh, ol = float(r["OR_HIGH"]), float(r["OR_LOW"])
                    c = float(r["Close"])
                    if side_str == "put" and c < ol:
                        entry_ts = ts
                        break
                    if side_str == "call" and c > oh:
                        entry_ts = ts
                        break
            else:
                continue

            if entry_ts is None:
                continue

            if htf_skip_counter:
                mtf = build_mtf_map(df, entry_ts)
                if side_str == "call" and mtf.bias_4h == "bear":
                    continue
                if side_str == "put" and mtf.bias_4h == "bull":
                    continue

            spot = float(closes.loc[entry_ts])
            entry = float(np.clip(spot * cfg.option_premium_pct_of_spot, 0.30, cfg.max_mid_price))
            entry *= 1 + cfg.spread_penalty_pct / 2
            if entry * 100 > cfg.max_notional_per_trade:
                continue

            # Optional: exit early if gap filled (prior close tagged) — still use TP/SL path
            exit_prem, reason = _simulate_0dte_path(
                closes,
                entry_ts,
                entry,
                spot,
                side,
                profit_target_pct=cfg.profit_target_pct,
                stop_loss_pct=cfg.stop_loss_pct,
                force_flat=force_flat,
                spread_penalty_pct=cfg.spread_penalty_pct,
            )
            # Also check fill-exit: if price tags prior close after entry, treat as soft target
            # (already handled loosely by TP; keep standard sim for apples-to-apples)

            pnl = (exit_prem - entry) * 100
            equity += pnl
            trades.append(
                {
                    "date": str(d),
                    "pnl": pnl,
                    "side": side_str,
                    "gap_pct": gap,
                    "entry": str(entry_ts),
                    "exit_reason": reason,
                }
            )
            equity_rows.append({"datetime": str(entry_ts), "equity": equity})
    finally:
        clear_htf_cache()

    tdf = pd.DataFrame(trades)
    curve = (
        pd.DataFrame(equity_rows)
        if equity_rows
        else pd.DataFrame([{"datetime": "x", "equity": equity}])
    )
    m = _metrics(tdf, curve, 100_000.0)
    return {
        "min_gap_pct": min_gap_pct,
        "max_gap_pct": max_gap_pct,
        "entry_mode": entry_mode,
        "require_early_toward": require_early_toward,
        "htf_skip_counter": htf_skip_counter,
        "n": m["n_trades"],
        "wr": m["win_rate"],
        "exp": m["expectancy"],
        "pf": m["profit_factor"],
        "pnl": float(tdf["pnl"].sum()) if len(tdf) else 0.0,
        "dd": m["max_drawdown"],
        "gate": bool(
            m["profit_factor"] >= cfg.backtest_gate_pf and m["n_trades"] >= cfg.backtest_gate_min_trades
        ),
    }


def main() -> None:
    bars = pd.read_parquet(ROOT / "data" / "bars" / "SPY_5m.parquet")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("America/New_York")
    df = enrich_bars(bars, orb_minutes=30)

    stats = gap_stats(df)
    print("=== Gap fill incidence (prior RTH close -> today open) ===")
    print(f"sessions: {len(stats)}")
    print(stats["gap_pct"].describe().to_string())
    print(f"\noverall fill rate: {stats['filled'].mean():.1%}")
    for thr in (0.001, 0.002, 0.003, 0.005):
        s = stats[stats["gap_abs"] >= thr]
        if len(s) == 0:
            continue
        print(f"|gap|>={thr*100:.1f}%  n={len(s):3d}  fill={s['filled'].mean():.1%}")
    up = stats[stats["gap_pct"] >= 0.002]
    dn = stats[stats["gap_pct"] <= -0.002]
    print(f"gap-up  >=0.2% n={len(up)} fill={up['filled'].mean():.1%}" if len(up) else "gap-up n=0")
    print(f"gap-dn  <=-0.2% n={len(dn)} fill={dn['filled'].mean():.1%}" if len(dn) else "gap-dn n=0")
    if stats["touch_time"].notna().any():
        print("median fill clock:", stats.loc[stats["filled"], "touch_time"].dropna().value_counts().head(5).to_string())

    variants = []
    for min_g, max_g in ((0.0015, 0.015), (0.002, 0.012), (0.003, 0.015)):
        for mode in ("open", "confirm", "orb_align"):
            for early in (False, True):
                for htf in (False, True):
                    if mode == "open" and early:
                        continue  # open entry ignores early filter timing
                    r = run_gap_fill_backtest(
                        df,
                        min_gap_pct=min_g,
                        max_gap_pct=max_g,
                        entry_mode=mode,
                        entry_latest=time(11, 0),
                        require_early_toward=early,
                        htf_skip_counter=htf,
                    )
                    variants.append(r)

    vdf = pd.DataFrame(variants).sort_values(["pf", "exp"], ascending=[False, False])
    print("\n=== 0DTE gap-fade backtests (top 12 by PF) ===")
    top = vdf.head(12)
    for _, r in top.iterrows():
        print(
            f"gap[{r['min_gap_pct']*100:.2f}-{r['max_gap_pct']*100:.1f}%] "
            f"{r['entry_mode']:10s} early={str(r['require_early_toward']):5s} "
            f"htf={str(r['htf_skip_counter']):5s} "
            f"n={r['n']:5.0f} WR={r['wr']:5.1%} E=${r['exp']:7.2f} "
            f"PF={r['pf']:5.2f} PnL=${r['pnl']:7.0f} gate={r['gate']}"
        )

    out = {
        "incidence": {
            "n_sessions": int(len(stats)),
            "overall_fill_rate": float(stats["filled"].mean()),
            "by_threshold": {
                f"{thr}": {
                    "n": int((stats["gap_abs"] >= thr).sum()),
                    "fill": float(stats.loc[stats["gap_abs"] >= thr, "filled"].mean())
                    if (stats["gap_abs"] >= thr).any()
                    else 0.0,
                }
                for thr in (0.001, 0.002, 0.003, 0.005)
            },
        },
        "variants": variants,
        "best": top.iloc[0].to_dict() if len(top) else {},
    }
    path = ROOT / "artifacts" / "gap_fill_research.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    stats.to_csv(ROOT / "artifacts" / "gap_fill_incidence.csv", index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
