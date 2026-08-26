"""ORB + 1H/4H filter research (bias / EQ location overlays)."""

from __future__ import annotations

import json
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from stockpro.spy_day.backtest import _metrics, _simulate_0dte_path
from stockpro.spy_day.mtf_liquidity import build_mtf_map, clear_htf_cache, set_htf_cache
from stockpro.spy_day.patterns import best_signal, enrich_bars
from stockpro.spy_day.session import load_spy_day_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    bars = pd.read_parquet(ROOT / "data" / "bars" / "SPY_5m.parquet")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("America/New_York")
    cfg = load_spy_day_config()
    df = enrich_bars(bars, orb_minutes=cfg.orb_minutes)
    set_htf_cache(df)

    force_flat = time(*map(int, cfg.force_flat_et.split(":")[:2]))
    late = time(*map(int, cfg.no_new_entries_after_et.split(":")[:2]))

    def run_filtered(name: str, allow_fn) -> dict:
        equity = 100_000.0
        trades: list[dict] = []
        equity_rows: list[dict] = []
        trades_today = 0
        current_day = None
        exit_until_idx = -1
        closes = df["Close"]
        for i in range(len(df)):
            ts = df.index[i]
            d = ts.date()
            if current_day != d:
                current_day = d
                trades_today = 0
            if i < exit_until_idx:
                continue
            t = ts.time()
            if t < time(9, 35) or t >= late:
                continue
            if trades_today >= cfg.max_trades_per_day:
                continue
            sig = best_signal(
                df,
                i,
                enabled=["orb"],
                min_confidence=0.70,
                pattern_min_confidence={"orb": 0.70},
            )
            if sig is None:
                continue
            mtf = build_mtf_map(df, ts)
            if not allow_fn(sig, mtf, df.iloc[i]):
                continue
            spot = float(sig.spot)
            entry = float(np.clip(spot * cfg.option_premium_pct_of_spot, 0.30, cfg.max_mid_price))
            entry *= 1 + cfg.spread_penalty_pct / 2
            if entry * 100 > cfg.max_notional_per_trade:
                continue
            side = 1 if sig.side == "call" else -1
            exit_prem, reason = _simulate_0dte_path(
                closes,
                ts,
                entry,
                spot,
                side,
                profit_target_pct=cfg.profit_target_pct,
                stop_loss_pct=cfg.stop_loss_pct,
                force_flat=force_flat,
                spread_penalty_pct=cfg.spread_penalty_pct,
            )
            pnl = (exit_prem - entry) * 100
            equity += pnl
            trades_today += 1
            loc = closes.index.get_loc(ts)
            exit_until_idx = i + 1
            if not isinstance(loc, slice):
                prem = entry
                for j in range(loc + 1, len(closes)):
                    prev = float(closes.iloc[j - 1])
                    cur = float(closes.iloc[j])
                    und_ret = (cur / prev - 1.0) * side if prev > 0 else 0.0
                    prem_frac = max(prem / spot, 1e-4)
                    prem *= 1.0 + float(np.clip(0.4 * und_ret / prem_frac, -0.55, 1.20))
                    ret = prem / entry - 1.0
                    exit_until_idx = j
                    if ret >= cfg.profit_target_pct or ret <= -cfg.stop_loss_pct:
                        break
                    if closes.index[j].time() >= force_flat or closes.index[j].date() != d:
                        break
            trades.append(
                {
                    "date": str(d),
                    "pnl": pnl,
                    "side": sig.side,
                    "bias": mtf.bias_4h,
                    "exit_reason": reason,
                }
            )
            equity_rows.append({"datetime": str(ts), "equity": equity})
        tdf = pd.DataFrame(trades)
        curve = (
            pd.DataFrame(equity_rows)
            if equity_rows
            else pd.DataFrame([{"datetime": "x", "equity": equity}])
        )
        m = _metrics(tdf, curve, 100_000.0)
        return {
            "name": name,
            "n": m["n_trades"],
            "wr": m["win_rate"],
            "exp": m["expectancy"],
            "pf": m["profit_factor"],
            "pnl": float(tdf["pnl"].sum()) if len(tdf) else 0.0,
            "dd": m["max_drawdown"],
            "gate": bool(
                m["profit_factor"] >= cfg.backtest_gate_pf
                and m["n_trades"] >= cfg.backtest_gate_min_trades
            ),
        }

    def with_trend(sig, mtf, row) -> bool:
        return (sig.side == "call" and mtf.bias_4h == "bull") or (
            sig.side == "put" and mtf.bias_4h == "bear"
        )

    def with_trend_or_neutral(sig, mtf, row) -> bool:
        if sig.side == "call":
            return mtf.bias_4h in ("bull", "neutral")
        return mtf.bias_4h in ("bear", "neutral")

    def skip_counter(sig, mtf, row) -> bool:
        if sig.side == "call" and mtf.bias_4h == "bear":
            return False
        if sig.side == "put" and mtf.bias_4h == "bull":
            return False
        return True

    def toward_eq(sig, mtf, row) -> bool:
        spot = float(row["Close"])
        if sig.side == "call":
            return mtf.nearest(spot, side="high", kinds=("eqh",), max_dist_pct=0.006, min_touches=2) is not None
        return mtf.nearest(spot, side="low", kinds=("eql",), max_dist_pct=0.006, min_touches=2) is not None

    def leave_eq(sig, mtf, row) -> bool:
        spot = float(row["Close"])
        if sig.side == "call":
            return mtf.nearest(spot, side="low", kinds=("eql", "eqh"), max_dist_pct=0.004, min_touches=2) is not None
        return mtf.nearest(spot, side="high", kinds=("eql", "eqh"), max_dist_pct=0.004, min_touches=2) is not None

    def trend_neut_and_eq(sig, mtf, row) -> bool:
        return with_trend_or_neutral(sig, mtf, row) and toward_eq(sig, mtf, row)

    rows = [
        run_filtered("orb_baseline", lambda s, m, r: True),
        run_filtered("orb_4h_with_trend", with_trend),
        run_filtered("orb_4h_trend_or_neutral", with_trend_or_neutral),
        run_filtered("orb_4h_skip_counter", skip_counter),
        run_filtered("orb_toward_1h4h_eq", toward_eq),
        run_filtered("orb_leave_eq_balance", leave_eq),
        run_filtered("orb_trend_neut+eq_target", trend_neut_and_eq),
    ]
    clear_htf_cache()

    for r in rows:
        print(
            f"{r['name']:28s} n={r['n']:5.0f} WR={r['wr']:5.1%} "
            f"E=${r['exp']:7.2f} PF={r['pf']:5.2f} PnL=${r['pnl']:7.0f} gate={r['gate']}"
        )
    out = ROOT / "artifacts" / "orb_htf_filter_research.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
