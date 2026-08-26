"""AMD vehicle research: same signals, three payoffs.

1) SPY 0DTE option proxy (current day-trade model)
2) SPY short-dated (~3 DTE) option proxy — multi-session, slower gamma, overnight theta
3) Futures-style points (MES $5/pt) — ATR stop/target, no theta

Answers whether AMD fails because of 0DTE decay or because the fade has no edge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from stockpro.spy_day.amd import AMD_PARAMS, amd_signal, configure_amd_params
from stockpro.spy_day.htf_permission import HtfPermissionConfig
from stockpro.spy_day.mtf_liquidity import clear_htf_cache, set_htf_cache
from stockpro.spy_day.patterns import configure_htf_permission, enrich_bars
from stockpro.spy_day.session import load_spy_day_config

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Sig:
    i: int
    ts: pd.Timestamp
    side: str  # call | put
    spot: float
    atr: float
    reason: str
    variant: str


def _pf(pnls: pd.Series) -> float:
    wins = pnls[pnls > 0].sum()
    losses = (-pnls[pnls < 0]).sum()
    if losses <= 1e-12:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def _stats(trades: list[dict]) -> dict:
    if not trades:
        return {
            "n": 0,
            "wr": 0.0,
            "exp": 0.0,
            "pf": 0.0,
            "pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }
    s = pd.Series([t["pnl"] for t in trades], dtype=float)
    wins = s[s > 0]
    losses = s[s <= 0]
    return {
        "n": int(len(s)),
        "wr": float((s > 0).mean()),
        "exp": float(s.mean()),
        "pf": _pf(s),
        "pnl": float(s.sum()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
    }


def _collect_signals(df: pd.DataFrame, variant: str, params: dict) -> list[Sig]:
    configure_amd_params(
        {
            "window_start": time(10, 0),
            "window_end": time(12, 0),
            "pierce_atr": 0.15,
            "one_per_day": True,
            "require_confirm": False,
            "apply_htf": False,
            "min_body_frac": 0.35,
        }
    )
    configure_amd_params(params)
    for k in ("window_start", "window_end"):
        if k in params:
            AMD_PARAMS[k] = params[k]

    cfg = load_spy_day_config()
    configure_htf_permission(
        HtfPermissionConfig(
            enabled=True,
            skip_4h_counter_trend=True,
            eq_context="none",
        )
        if params.get("apply_htf")
        else HtfPermissionConfig(enabled=False)
    )

    out: list[Sig] = []
    last_day = None
    for i in range(len(df)):
        ts = df.index[i]
        d = ts.date()
        if last_day != d:
            last_day = d
        sig = amd_signal(df, i)
        if not sig:
            continue
        # one collected signal per day (detector also one_per_day)
        if out and out[-1].ts.date() == d:
            continue
        atr = float(df.iloc[i]["ATR"]) if pd.notna(df.iloc[i].get("ATR")) else float("nan")
        out.append(
            Sig(
                i=i,
                ts=ts,
                side=sig.side,
                spot=float(sig.spot),
                atr=atr,
                reason=sig.reason,
                variant=variant,
            )
        )
    return out


def sim_0dte(df: pd.DataFrame, sig: Sig) -> dict:
    """Same-session 0DTE premium proxy (matches spy_day backtest economics)."""
    side = 1 if sig.side == "call" else -1
    entry = float(np.clip(sig.spot * 0.002, 0.30, 3.0)) * 1.02
    force_flat = time(15, 45)
    closes = df["Close"]
    loc = sig.i
    prem = entry
    reason = "eod"
    for j in range(loc + 1, len(df)):
        ts = df.index[j]
        prev, cur = float(closes.iloc[j - 1]), float(closes.iloc[j])
        if prev <= 0:
            continue
        und_ret = (cur / prev - 1.0) * side
        prem_frac = max(prem / sig.spot, 1e-4)
        prem *= 1.0 + float(np.clip(0.40 * und_ret / prem_frac, -0.55, 1.20))
        ret = prem / entry - 1.0
        if ret >= 0.30:
            prem *= 0.98
            reason = "profit_target"
            break
        if ret <= -0.25:
            prem *= 0.98
            reason = "stop_loss"
            break
        if ts.time() >= force_flat or ts.date() != sig.ts.date():
            prem *= 0.98
            reason = "force_flat"
            break
    pnl = (prem - entry) * 100
    return {"vehicle": "0dte", "pnl": pnl, "exit_reason": reason, "entry": entry, "exit": prem}


def sim_short_dte(df: pd.DataFrame, sig: Sig, *, max_sessions: int = 3) -> dict:
    """~3 DTE ATM-ish proxy: slower gamma, overnight theta, multi-session hold."""
    side = 1 if sig.side == "call" else -1
    entry = float(np.clip(sig.spot * 0.008, 0.80, 4.0)) * 1.02  # richer than 0DTE
    closes = df["Close"]
    prem = entry
    reason = "time_exit"
    sessions_seen = {sig.ts.date()}
    last_date = sig.ts.date()
    # Dampen 0DTE-style mark-to-market (longer dated moves slower vs premium)
    gamma_scale = 0.55
    overnight_theta = 0.03  # ~3% premium haircut each overnight

    for j in range(sig.i + 1, len(df)):
        ts = df.index[j]
        d = ts.date()
        if d != last_date:
            prem *= 1.0 - overnight_theta
            sessions_seen.add(d)
            last_date = d
            if len(sessions_seen) > max_sessions:
                reason = "max_sessions"
                break
        prev, cur = float(closes.iloc[j - 1]), float(closes.iloc[j])
        if prev <= 0:
            continue
        und_ret = (cur / prev - 1.0) * side
        prem_frac = max(prem / sig.spot, 1e-4)
        prem *= 1.0 + float(np.clip(gamma_scale * 0.40 * und_ret / prem_frac, -0.35, 0.80))
        ret = prem / entry - 1.0
        if ret >= 0.35:
            prem *= 0.98
            reason = "profit_target"
            break
        if ret <= -0.30:
            prem *= 0.98
            reason = "stop_loss"
            break
        # Flatten late on last allowed session
        if len(sessions_seen) >= max_sessions and ts.time() >= time(15, 45):
            prem *= 0.98
            reason = "force_flat"
            break
    pnl = (prem - entry) * 100
    return {
        "vehicle": "short_dte_3d",
        "pnl": pnl,
        "exit_reason": reason,
        "entry": entry,
        "exit": prem,
    }


def sim_futures_points(df: pd.DataFrame, sig: Sig, *, point_value: float = 5.0) -> dict:
    """MES-style: $5 per SPY/ES point, 1.5R target / 1R stop, same-day force flat."""
    if not (sig.atr == sig.atr) or sig.atr <= 0:
        return {"vehicle": "futures_mes", "pnl": 0.0, "exit_reason": "no_atr", "entry": sig.spot, "exit": sig.spot}
    side = 1 if sig.side == "call" else -1
    entry = sig.spot
    stop_dist = 1.0 * sig.atr
    tgt_dist = 1.5 * sig.atr
    stop = entry - side * stop_dist
    target = entry + side * tgt_dist
    reason = "force_flat"
    exit_px = entry
    for j in range(sig.i + 1, len(df)):
        ts = df.index[j]
        if ts.date() != sig.ts.date():
            exit_px = float(df.iloc[j - 1]["Close"])
            reason = "session_end"
            break
        hi, lo, c = float(df.iloc[j]["High"]), float(df.iloc[j]["Low"]), float(df.iloc[j]["Close"])
        # Conservative: stop before target if both could be touched
        if side > 0:
            hit_stop = lo <= stop
            hit_tgt = hi >= target
        else:
            hit_stop = hi >= stop
            hit_tgt = lo <= target
        if hit_stop and hit_tgt:
            exit_px = stop
            reason = "stop_loss"
            break
        if hit_stop:
            exit_px = stop
            reason = "stop_loss"
            break
        if hit_tgt:
            exit_px = target
            reason = "profit_target"
            break
        if ts.time() >= time(15, 45):
            exit_px = c
            reason = "force_flat"
            break
        exit_px = c
    points = (exit_px - entry) * side
    pnl = points * point_value
    return {
        "vehicle": "futures_mes",
        "pnl": float(pnl),
        "exit_reason": reason,
        "entry": entry,
        "exit": exit_px,
        "points": float(points),
    }


def sim_futures_hold(df: pd.DataFrame, sig: Sig, *, point_value: float = 5.0, max_sessions: int = 2) -> dict:
    """Same ATR exits but allow overnight (closer to swing AMD on futures)."""
    if not (sig.atr == sig.atr) or sig.atr <= 0:
        return {
            "vehicle": "futures_mes_hold",
            "pnl": 0.0,
            "exit_reason": "no_atr",
            "entry": sig.spot,
            "exit": sig.spot,
        }
    side = 1 if sig.side == "call" else -1
    entry = sig.spot
    stop = entry - side * 1.0 * sig.atr
    target = entry + side * 1.5 * sig.atr
    reason = "max_sessions"
    exit_px = entry
    sessions = {sig.ts.date()}
    for j in range(sig.i + 1, len(df)):
        ts = df.index[j]
        sessions.add(ts.date())
        hi, lo, c = float(df.iloc[j]["High"]), float(df.iloc[j]["Low"]), float(df.iloc[j]["Close"])
        if side > 0:
            hit_stop, hit_tgt = lo <= stop, hi >= target
        else:
            hit_stop, hit_tgt = hi >= stop, lo <= target
        if hit_stop:
            exit_px, reason = stop, "stop_loss"
            break
        if hit_tgt:
            exit_px, reason = target, "profit_target"
            break
        if len(sessions) > max_sessions and ts.time() >= time(15, 45):
            exit_px, reason = c, "max_sessions"
            break
        exit_px = c
    points = (exit_px - entry) * side
    return {
        "vehicle": "futures_mes_hold",
        "pnl": float(points * point_value),
        "exit_reason": reason,
        "entry": entry,
        "exit": exit_px,
        "points": float(points),
    }


VARIANTS = [
    ("amd_reclaim_am", {"require_confirm": False, "apply_htf": False}),
    ("amd_reclaim_htf", {"require_confirm": False, "apply_htf": True}),
    ("amd_confirm_am", {"require_confirm": True, "apply_htf": False}),
]


def main() -> None:
    bars = pd.read_parquet(ROOT / "data" / "bars" / "SPY_5m.parquet")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("America/New_York")
    df = enrich_bars(bars, orb_minutes=30)
    set_htf_cache(df)

    print("=== AMD vehicle research (same signals, different payoffs) ===")
    print(f"bars={len(df)}  {df.index.min()} -> {df.index.max()}\n")
    print("Vehicles:")
    print("  0dte          — same-day option proxy (live stack economics)")
    print("  short_dte_3d  — ~3 DTE option proxy, overnight theta, max 3 sessions")
    print("  futures_mes   — $5/pt, 1.5R/1R ATR, same-day flat")
    print("  futures_mes_hold — $5/pt ATR exits, hold up to 2 sessions\n")

    summary_rows: list[dict] = []
    all_trades: list[dict] = []

    try:
        for vname, params in VARIANTS:
            print(f"--- Signals: {vname} ---", flush=True)
            sigs = _collect_signals(df, vname, params)
            print(f"  signals={len(sigs)}", flush=True)
            by_vehicle: dict[str, list[dict]] = {
                "0dte": [],
                "short_dte_3d": [],
                "futures_mes": [],
                "futures_mes_hold": [],
            }
            for sig in sigs:
                for sim in (sim_0dte, sim_short_dte, sim_futures_points, sim_futures_hold):
                    tr = sim(df, sig)
                    tr.update(
                        {
                            "variant": vname,
                            "datetime": str(sig.ts),
                            "side": sig.side,
                            "spot": sig.spot,
                            "reason": sig.reason,
                        }
                    )
                    by_vehicle[tr["vehicle"]].append(tr)
                    all_trades.append(tr)

            for veh, trades in by_vehicle.items():
                st = _stats(trades)
                row = {"variant": vname, "vehicle": veh, **st}
                summary_rows.append(row)
                print(
                    f"  {veh:<18} n={st['n']:<3} WR={st['wr']*100:5.1f}% "
                    f"PF={st['pf']:.2f} exp=${st['exp']:.2f} pnl=${st['pnl']:.0f}"
                )
            print()
    finally:
        clear_htf_cache()
        configure_amd_params(
            {
                "require_confirm": False,
                "apply_htf": False,
                "window_start": time(10, 0),
                "window_end": time(12, 0),
            }
        )

    # Ranking: futures / short_dte vs 0dte
    print("--- Ranked by PF (n>=20) ---")
    ranked = sorted(
        [r for r in summary_rows if r["n"] >= 20],
        key=lambda r: (r["pf"] if r["pf"] == r["pf"] else 0.0, r["exp"]),
        reverse=True,
    )
    for r in ranked:
        print(
            f"{r['variant']:<18} {r['vehicle']:<18} n={r['n']} "
            f"WR={r['wr']*100:.1f}% PF={r['pf']:.2f} pnl=${r['pnl']:.0f}"
        )

    out_csv = ROOT / "artifacts" / "amd_vehicles_research.csv"
    out_json = ROOT / "artifacts" / "amd_vehicles_research.json"
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps({"summary": summary_rows, "n_trades": len(all_trades)}, indent=2), encoding="utf-8")
    pd.DataFrame(all_trades).to_csv(ROOT / "artifacts" / "amd_vehicles_trades.csv", index=False)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
