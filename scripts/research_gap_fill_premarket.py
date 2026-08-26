"""6-month gap-fill research using true Alpaca premarket bars."""

from __future__ import annotations

import json
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from stockpro.data.bars_5m import filter_premarket, filter_rth, load_cached_bars_5m_ext
from stockpro.spy_day.backtest import _metrics, _simulate_0dte_path
from stockpro.spy_day.mtf_liquidity import build_mtf_map, clear_htf_cache, set_htf_cache
from stockpro.spy_day.patterns import enrich_bars
from stockpro.spy_day.session import load_spy_day_config

ROOT = Path(__file__).resolve().parents[1]


def build_gap_table(ext: pd.DataFrame, rth: pd.DataFrame) -> pd.DataFrame:
    """Per session: overnight gap + premarket path + whether gap still open at the bell."""
    rows = []
    days = sorted({ts.date() for ts in rth.index})
    pm_all = filter_premarket(ext)
    for i, d in enumerate(days):
        if i == 0:
            continue
        day = rth[rth.index.date == d]
        prev = rth[rth.index.date == days[i - 1]]
        if len(day) < 10 or len(prev) < 5:
            continue
        rth_open = float(day.iloc[0]["Open"])
        prior_close = float(prev.iloc[-1]["Close"])
        gap = (rth_open - prior_close) / prior_close
        pm = pm_all[pm_all.index.date == d]
        has_pm = len(pm) > 0
        if has_pm:
            pm_open = float(pm.iloc[0]["Open"])
            pm_high = float(pm["High"].max())
            pm_low = float(pm["Low"].min())
            pm_last = float(pm.iloc[-1]["Close"])
            filled_in_pm = pm_low <= prior_close <= pm_high
            # Premarket moved toward fill from its open vs prior close
            pm_gap = (pm_open - prior_close) / prior_close if prior_close else 0.0
            pm_progress = 0.0
            if abs(gap) > 1e-9:
                # how much of the RTH gap was closed by pm_last
                if gap > 0:
                    pm_progress = (rth_open - pm_last) / (rth_open - prior_close) if rth_open != prior_close else 0.0
                else:
                    pm_progress = (pm_last - rth_open) / (prior_close - rth_open) if rth_open != prior_close else 0.0
        else:
            pm_open = pm_high = pm_low = pm_last = float("nan")
            filled_in_pm = False
            pm_gap = float("nan")
            pm_progress = float("nan")

        day_hi = float(day["High"].max())
        day_lo = float(day["Low"].min())
        filled_rth = day_lo <= prior_close <= day_hi
        remaining_gap = (rth_open - prior_close) / prior_close
        # Still open at bell = not filled in PM (or no PM data)
        still_open = (not filled_in_pm) if has_pm else True

        rows.append(
            {
                "date": str(d),
                "gap_pct": gap,
                "gap_abs": abs(gap),
                "prior_close": prior_close,
                "rth_open": rth_open,
                "has_pm": has_pm,
                "pm_bars": int(len(pm)),
                "pm_open": pm_open,
                "pm_high": pm_high,
                "pm_low": pm_low,
                "pm_last": pm_last,
                "pm_gap": pm_gap,
                "filled_in_pm": filled_in_pm,
                "still_open_at_bell": still_open,
                "filled_rth": filled_rth,
                "pm_progress": pm_progress,
            }
        )
    return pd.DataFrame(rows)


def run_variant(
    rth: pd.DataFrame,
    gaps: pd.DataFrame,
    *,
    min_gap_pct: float,
    max_gap_pct: float,
    entry_mode: str,
    require_still_open: bool,
    require_pm: bool,
    htf_skip_counter: bool,
    label: str,
) -> dict:
    cfg = load_spy_day_config()
    force_flat = time(*map(int, cfg.force_flat_et.split(":")[:2]))
    set_htf_cache(rth)
    equity = 100_000.0
    trades: list[dict] = []
    equity_rows: list[dict] = []
    closes = rth["Close"]
    gap_by_date = {r["date"]: r for _, r in gaps.iterrows()}

    try:
        for d in sorted({ts.date() for ts in rth.index}):
            key = str(d)
            g = gap_by_date.get(key)
            if g is None:
                continue
            gap = float(g["gap_pct"])
            gap_abs = abs(gap)
            if gap_abs < min_gap_pct or gap_abs > max_gap_pct:
                continue
            if require_pm and not bool(g["has_pm"]):
                continue
            if require_still_open and not bool(g["still_open_at_bell"]):
                continue

            side_str = "put" if gap > 0 else "call"
            side = -1 if side_str == "put" else 1
            day = rth[rth.index.date == d]
            if len(day) < 15:
                continue
            o = float(day.iloc[0]["Open"])

            entry_ts = None
            if entry_mode == "open":
                entry_ts = day.index[0]
            elif entry_mode == "confirm":
                for ts, row in day.iterrows():
                    if ts.time() < time(9, 35) or ts.time() > time(11, 0):
                        continue
                    c = float(row["Close"])
                    if gap > 0 and c < o:
                        entry_ts = ts
                        break
                    if gap < 0 and c > o:
                        entry_ts = ts
                        break
            elif entry_mode == "orb_align":
                for ts, row in day.iterrows():
                    if not bool(row.get("OR_READY", False)):
                        continue
                    if ts.time() > time(11, 0):
                        break
                    oh, ol = float(row["OR_HIGH"]), float(row["OR_LOW"])
                    c = float(row["Close"])
                    if side_str == "put" and c < ol:
                        entry_ts = ts
                        break
                    if side_str == "call" and c > oh:
                        entry_ts = ts
                        break
            if entry_ts is None:
                continue

            if htf_skip_counter:
                mtf = build_mtf_map(rth, entry_ts)
                if side_str == "call" and mtf.bias_4h == "bear":
                    continue
                if side_str == "put" and mtf.bias_4h == "bull":
                    continue

            spot = float(closes.loc[entry_ts])
            entry = float(np.clip(spot * cfg.option_premium_pct_of_spot, 0.30, cfg.max_mid_price))
            entry *= 1 + cfg.spread_penalty_pct / 2
            if entry * 100 > cfg.max_notional_per_trade:
                continue
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
            pnl = (exit_prem - entry) * 100
            equity += pnl
            trades.append(
                {
                    "date": key,
                    "pnl": pnl,
                    "side": side_str,
                    "gap_pct": gap,
                    "filled_in_pm": bool(g["filled_in_pm"]),
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
        "label": label,
        "entry_mode": entry_mode,
        "min_gap_pct": min_gap_pct,
        "require_still_open": require_still_open,
        "require_pm": require_pm,
        "htf_skip_counter": htf_skip_counter,
        "n": m["n_trades"],
        "wr": m["win_rate"],
        "exp": m["expectancy"],
        "pf": m["profit_factor"],
        "pnl": float(tdf["pnl"].sum()) if len(tdf) else 0.0,
        "gate": bool(
            m["profit_factor"] >= cfg.backtest_gate_pf and m["n_trades"] >= cfg.backtest_gate_min_trades
        ),
    }


def main() -> None:
    ext = load_cached_bars_5m_ext()
    if ext.empty:
        raise SystemExit("No extended bars cache. Run: python scripts/refresh_spy_bars.py --full --days 180")

    rth = enrich_bars(filter_rth(ext), orb_minutes=30)
    gaps = build_gap_table(ext, rth)
    pm_days = int(gaps["has_pm"].sum()) if len(gaps) else 0

    print("=== Premarket coverage ===")
    print(f"ext bars: {len(ext)}  {ext.index.min()} -> {ext.index.max()}")
    print(f"RTH bars: {len(rth)}")
    print(f"sessions: {len(gaps)}  with premarket bars: {pm_days} ({pm_days/max(len(gaps),1):.0%})")

    with_pm = gaps[gaps["has_pm"]]
    print("\n=== Gap fill incidence (with PM filter where available) ===")
    print(f"overall RTH fill: {gaps['filled_rth'].mean():.1%}")
    if len(with_pm):
        print(f"filled already in premarket: {with_pm['filled_in_pm'].mean():.1%}")
        still = with_pm[with_pm["still_open_at_bell"] & (with_pm["gap_abs"] >= 0.002)]
        print(
            f"still-open gaps >=0.2% at bell: n={len(still)} "
            f"RTH fill={still['filled_rth'].mean():.1%}" if len(still) else "still-open n=0"
        )
        already = with_pm[with_pm["filled_in_pm"] & (with_pm["gap_abs"] >= 0.002)]
        print(
            f"already-filled-in-PM gaps >=0.2%: n={len(already)} "
            f"(skip these for fade)" if len(already) else "already-filled n=0"
        )

    # Baseline: RTH-only style (ignore PM) vs PM-aware
    variants = []
    for mode in ("open", "confirm", "orb_align"):
        for min_g in (0.0015, 0.002, 0.003):
            variants.append(
                run_variant(
                    rth,
                    gaps,
                    min_gap_pct=min_g,
                    max_gap_pct=0.015,
                    entry_mode=mode,
                    require_still_open=False,
                    require_pm=False,
                    htf_skip_counter=False,
                    label=f"rth_proxy|{mode}|g>={min_g}",
                )
            )
            variants.append(
                run_variant(
                    rth,
                    gaps,
                    min_gap_pct=min_g,
                    max_gap_pct=0.015,
                    entry_mode=mode,
                    require_still_open=True,
                    require_pm=True,
                    htf_skip_counter=False,
                    label=f"pm_still_open|{mode}|g>={min_g}",
                )
            )
            variants.append(
                run_variant(
                    rth,
                    gaps,
                    min_gap_pct=min_g,
                    max_gap_pct=0.015,
                    entry_mode=mode,
                    require_still_open=True,
                    require_pm=True,
                    htf_skip_counter=True,
                    label=f"pm_still_open+htf|{mode}|g>={min_g}",
                )
            )

    vdf = pd.DataFrame(variants).sort_values(["wr", "pf", "exp"], ascending=[False, False, False])
    print("\n=== Top by win rate ===")
    for _, r in vdf.head(15).iterrows():
        print(
            f"{r['label'][:42]:42s} n={r['n']:5.0f} WR={r['wr']:5.1%} "
            f"E=${r['exp']:7.2f} PF={r['pf']:5.2f} PnL=${r['pnl']:6.0f} gate={r['gate']}"
        )

    # Head-to-head: best orb_align RTH proxy vs PM still-open
    print("\n=== Head-to-head orb_align gap>=0.15% ===")
    for lab in (
        "rth_proxy|orb_align|g>=0.0015",
        "pm_still_open|orb_align|g>=0.0015",
        "pm_still_open+htf|orb_align|g>=0.0015",
    ):
        row = vdf[vdf["label"] == lab]
        if len(row):
            r = row.iloc[0]
            print(
                f"{lab:40s} n={r['n']:.0f} WR={r['wr']:.1%} E=${r['exp']:.2f} PF={r['pf']:.2f}"
            )

    out = {
        "coverage": {
            "ext_bars": int(len(ext)),
            "rth_bars": int(len(rth)),
            "sessions": int(len(gaps)),
            "sessions_with_pm": pm_days,
            "range": [str(ext.index.min()), str(ext.index.max())],
        },
        "incidence": {
            "rth_fill_rate": float(gaps["filled_rth"].mean()) if len(gaps) else 0.0,
            "pm_fill_rate": float(with_pm["filled_in_pm"].mean()) if len(with_pm) else 0.0,
        },
        "variants": variants,
        "best_wr": vdf.iloc[0].to_dict() if len(vdf) else {},
    }
    path = ROOT / "artifacts" / "gap_fill_premarket_research.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    gaps.to_csv(ROOT / "artifacts" / "gap_fill_premarket_incidence.csv", index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
