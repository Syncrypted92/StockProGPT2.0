"""SPY day backtest on 5m patterns — 0DTE ORB family + short-DTE AMD proxy."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from stockpro.spy_day.amd import configure_amd_params
from stockpro.spy_day.mtf_liquidity import clear_htf_cache, set_htf_cache
from stockpro.spy_day.ote import build_ote_cache, clear_ote_cache
from stockpro.spy_day.patterns import (
    best_signal,
    clear_ote_day_state,
    configure_htf_permission,
    configure_ote_params,
    enrich_bars,
)
from stockpro.spy_day.session import SpyDayConfig, load_spy_day_config

ET = ZoneInfo("America/New_York")


@dataclass
class SpyDayBacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict[str, float]
    by_pattern: dict[str, dict[str, float]]


@dataclass
class ScaleOutConfig:
    """Scale-out: bank lot 1 at TP1; remaining lots target tp2 and/or BE.

    qty=3 (live default): lot1 TP1, lot2 +45% with original SL, lot3 runner trails after TP2.
    qty=2: lot1 TP1, lot2 +45% and SL→entry after TP1.
    """

    qty: int = 3
    tp1_pct: float | None = None  # None = use coded 0DTE/AMD TP
    tp2_pct: float = 0.45
    tp3_pct: float | None = None  # None = no hard TP on runner
    runner_trail_pct: float | None = None  # after TP2, trail this far off runner peak
    runner_stop_at_entry: bool = True
    qty_by_pattern: dict[str, int] = field(default_factory=dict)

    def qty_for(self, pattern: str) -> int:
        if pattern in self.qty_by_pattern:
            return max(1, int(self.qty_by_pattern[pattern]))
        return max(1, int(self.qty))

    def for_pattern(self, pattern: str) -> "ScaleOutConfig":
        return ScaleOutConfig(
            qty=self.qty_for(pattern),
            tp1_pct=self.tp1_pct,
            tp2_pct=self.tp2_pct,
            tp3_pct=self.tp3_pct,
            runner_trail_pct=self.runner_trail_pct,
            runner_stop_at_entry=self.runner_stop_at_entry,
            qty_by_pattern=dict(self.qty_by_pattern),
        )


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(dd.min()) if len(dd) else 0.0


def _simulate_0dte_path(
    closes: pd.Series,
    entry_ts: pd.Timestamp,
    entry_prem: float,
    spot: float,
    side: int,
    *,
    profit_target_pct: float,
    stop_loss_pct: float,
    force_flat: time,
    spread_penalty_pct: float,
    delta: float = 0.40,
    tp_unlock_idx: int | None = None,
) -> tuple[float, str, int]:
    """Walk forward 5m closes until TP/SL/force-flat. Returns (exit_prem, reason, exit_idx)."""
    if entry_ts not in closes.index:
        return entry_prem, "missing", -1
    loc = closes.index.get_loc(entry_ts)
    if isinstance(loc, slice):
        return entry_prem, "missing", -1
    prem = entry_prem
    exit_j = int(loc)
    for j in range(loc + 1, len(closes)):
        exit_j = j
        ts = closes.index[j]
        prev = float(closes.iloc[j - 1])
        cur = float(closes.iloc[j])
        if prev <= 0:
            continue
        und_ret = (cur / prev - 1.0) * side
        prem_frac = max(prem / spot, 1e-4)
        day_opt = float(np.clip(delta * und_ret / prem_frac, -0.55, 1.20))
        prem = prem * (1.0 + day_opt)
        ret = prem / entry_prem - 1.0
        tp_ok = tp_unlock_idx is None or j >= tp_unlock_idx
        if tp_ok and ret >= profit_target_pct:
            return prem * (1 - spread_penalty_pct / 2), "profit_target", j
        if ret <= -stop_loss_pct:
            return prem * (1 - spread_penalty_pct / 2), "stop_loss", j
        t = ts.timetz().replace(tzinfo=None) if hasattr(ts, "timetz") else ts.time()
        if t >= force_flat:
            return prem * (1 - spread_penalty_pct / 2), "force_flat", j
        if ts.date() != entry_ts.date():
            return prem * (1 - spread_penalty_pct / 2), "session_end", j
    return prem * (1 - spread_penalty_pct / 2), "eod", exit_j


def _simulate_short_dte_path(
    closes: pd.Series,
    entry_ts: pd.Timestamp,
    entry_prem: float,
    spot: float,
    side: int,
    *,
    profit_target_pct: float,
    stop_loss_pct: float,
    max_sessions: int = 3,
    gamma_scale: float = 0.55,
    overnight_theta: float = 0.03,
    spread_penalty_pct: float = 0.04,
) -> tuple[float, str, int]:
    """Multi-session short-dated option proxy (AMD paper lane). No same-day force flat."""
    if entry_ts not in closes.index:
        return entry_prem, "missing", -1
    loc = closes.index.get_loc(entry_ts)
    if isinstance(loc, slice):
        return entry_prem, "missing", -1
    prem = entry_prem
    sessions_seen = {entry_ts.date()}
    last_date = entry_ts.date()
    exit_j = int(loc)
    for j in range(int(loc) + 1, len(closes)):
        exit_j = j
        ts = closes.index[j]
        d = ts.date()
        if d != last_date:
            prem *= 1.0 - overnight_theta
            sessions_seen.add(d)
            last_date = d
            if len(sessions_seen) > max_sessions:
                return prem * (1 - spread_penalty_pct / 2), "max_sessions", j
        prev = float(closes.iloc[j - 1])
        cur = float(closes.iloc[j])
        if prev <= 0:
            continue
        und_ret = (cur / prev - 1.0) * side
        prem_frac = max(prem / spot, 1e-4)
        prem *= 1.0 + float(np.clip(gamma_scale * 0.40 * und_ret / prem_frac, -0.35, 0.80))
        ret = prem / entry_prem - 1.0
        if ret >= profit_target_pct:
            return prem * (1 - spread_penalty_pct / 2), "profit_target", j
        if ret <= -stop_loss_pct:
            return prem * (1 - spread_penalty_pct / 2), "stop_loss", j
        t = ts.timetz().replace(tzinfo=None) if hasattr(ts, "timetz") else ts.time()
        if len(sessions_seen) >= max_sessions and t >= time(15, 45):
            return prem * (1 - spread_penalty_pct / 2), "time_stop", j
    return prem * (1 - spread_penalty_pct / 2), "eod", exit_j


def _bar_clock(ts: pd.Timestamp) -> time:
    return ts.timetz().replace(tzinfo=None) if hasattr(ts, "timetz") else ts.time()


def _iter_premium_path(
    closes: pd.Series,
    entry_ts: pd.Timestamp,
    entry_prem: float,
    spot: float,
    side: int,
    *,
    short_dte: bool,
    force_flat: time,
    spread_penalty_pct: float,
    max_sessions: int = 3,
    gamma_scale: float = 0.55,
    overnight_theta: float = 0.03,
    delta: float = 0.40,
) -> list[tuple[int, float, float, str | None]]:
    """Bar-by-bar option proxy. Each row: (idx, prem, ret, time_exit_or_none)."""
    if entry_ts not in closes.index:
        return []
    loc = closes.index.get_loc(entry_ts)
    if isinstance(loc, slice):
        return []
    prem = entry_prem
    sessions_seen = {entry_ts.date()}
    last_date = entry_ts.date()
    out: list[tuple[int, float, float, str | None]] = []
    for j in range(int(loc) + 1, len(closes)):
        ts = closes.index[j]
        time_exit: str | None = None
        if short_dte:
            d = ts.date()
            if d != last_date:
                prem *= 1.0 - overnight_theta
                sessions_seen.add(d)
                last_date = d
                if len(sessions_seen) > max_sessions:
                    time_exit = "max_sessions"
        prev = float(closes.iloc[j - 1])
        cur = float(closes.iloc[j])
        if prev > 0:
            und_ret = (cur / prev - 1.0) * side
            prem_frac = max(prem / spot, 1e-4)
            if short_dte:
                prem *= 1.0 + float(
                    np.clip(gamma_scale * 0.40 * und_ret / prem_frac, -0.35, 0.80)
                )
            else:
                prem = prem * (1.0 + float(np.clip(delta * und_ret / prem_frac, -0.55, 1.20)))
        ret = prem / entry_prem - 1.0
        t = _bar_clock(ts)
        if time_exit is None:
            if short_dte:
                if len(sessions_seen) >= max_sessions and t >= time(15, 45):
                    time_exit = "time_stop"
            else:
                if t >= force_flat:
                    time_exit = "force_flat"
                elif ts.date() != entry_ts.date():
                    time_exit = "session_end"
        out.append((j, prem, ret, time_exit))
        if time_exit in {"max_sessions"}:
            break
    if out and out[-1][3] is None:
        j, prem, ret, _ = out[-1]
        out[-1] = (j, prem, ret, "eod")
    return out


def _exit_px(prem: float, spread_penalty_pct: float) -> float:
    return prem * (1 - spread_penalty_pct / 2)


def simulate_scale_out(
    closes: pd.Series,
    entry_ts: pd.Timestamp,
    entry_prem: float,
    spot: float,
    side: int,
    *,
    profit_target_pct: float,
    stop_loss_pct: float,
    short_dte: bool,
    force_flat: time,
    spread_penalty_pct: float,
    scale: ScaleOutConfig | None = None,
    tp_unlock_idx: int | None = None,
    amd_flat_if_no_tp1: bool = False,
    amd_force_flat: bool = False,
) -> tuple[float, str, int, list[dict[str, Any]]]:
    """Scale-out on the 5m premium path.

    Before TP1 all lots share the original SL / time stop. After TP1:
    - qty=2: remaining lot targets tp2; SL moves to entry if runner_stop_at_entry
    - qty=3: lot 2 targets tp2 with original SL; lot 3 runner SL→entry
    amd_force_flat: short-DTE always flats at force_flat same day (no overnight).
    amd_flat_if_no_tp1: short-DTE flats at force_flat only before TP1.
    """
    scale = scale or ScaleOutConfig()
    n = max(1, int(scale.qty))
    tp1 = float(scale.tp1_pct) if scale.tp1_pct is not None else float(profit_target_pct)
    steps = _iter_premium_path(
        closes,
        entry_ts,
        entry_prem,
        spot,
        side,
        short_dte=short_dte,
        force_flat=force_flat,
        spread_penalty_pct=spread_penalty_pct,
    )
    if not steps:
        return entry_prem, "missing", -1, []

    open_lots = set(range(1, n + 1))
    tp2_lots = {2} if n >= 2 else set()
    if n == 2:
        be_lots = {2} if scale.runner_stop_at_entry else set()
        sl_lots = {2} if not scale.runner_stop_at_entry else set()
    else:
        be_lots = {n} if scale.runner_stop_at_entry and n >= 3 else set()
        sl_lots = (set(range(2, n + 1)) - be_lots) if n >= 2 else set()

    armed = False
    peak_ret = 0.0
    legs: list[dict[str, Any]] = []
    last_j = steps[0][0]
    sp = spread_penalty_pct

    def close_lot(lot: int, prem: float, reason: str, j: int, ret: float, px: float | None = None) -> None:
        nonlocal last_j
        if lot not in open_lots:
            return
        open_lots.remove(lot)
        last_j = j
        exit_p = px if px is not None else _exit_px(prem, sp)
        legs.append(
            {
                "lot": lot,
                "exit_reason": reason,
                "exit": exit_p,
                "ret": (exit_p / entry_prem - 1.0),
                "pnl": (exit_p - entry_prem) * 100,
            }
        )

    for j, prem, ret, time_exit in steps:
        last_j = j
        ts = closes.index[j]
        same_day_flat = (
            short_dte
            and ts.date() == entry_ts.date()
            and _bar_clock(ts) >= force_flat
            and (amd_force_flat or (amd_flat_if_no_tp1 and not armed))
        )
        if not armed:
            tp_ok = tp_unlock_idx is None or j >= tp_unlock_idx
            if tp_ok and ret >= tp1:
                close_lot(1, prem, "profit_target", j, ret)
                armed = True
            elif ret <= -stop_loss_pct:
                for lot in list(open_lots):
                    close_lot(lot, prem, "stop_loss", j, ret)
                break
            elif same_day_flat:
                for lot in list(open_lots):
                    close_lot(lot, prem, "session_flat", j, ret)
                break
            elif time_exit:
                for lot in list(open_lots):
                    close_lot(lot, prem, time_exit, j, ret)
                break
            else:
                continue

        if armed and open_lots:
            if same_day_flat:
                for lot in list(open_lots):
                    close_lot(lot, prem, "session_flat", j, ret)
                break
            peak_ret = max(peak_ret, ret)
            tp2_still_open = bool(open_lots & tp2_lots)
            for lot in list(open_lots):
                tp_ok = tp_unlock_idx is None or j >= tp_unlock_idx
                if lot in tp2_lots and tp_ok and ret >= scale.tp2_pct:
                    close_lot(lot, prem, "profit_target_2", j, ret)
                    continue
                if (
                    lot in be_lots
                    and scale.tp3_pct is not None
                    and tp_ok
                    and ret >= float(scale.tp3_pct)
                ):
                    close_lot(lot, prem, "profit_target_3", j, ret)
                    continue
                if (
                    lot in be_lots
                    and scale.runner_trail_pct
                    and not tp2_still_open
                    and (peak_ret - ret) >= float(scale.runner_trail_pct) - 1e-12
                ):
                    close_lot(lot, prem, "runner_trail", j, ret)
                    continue
                if lot in be_lots and ret <= 0.0:
                    close_lot(lot, prem, "breakeven", j, ret, px=entry_prem)
                    continue
                if lot in sl_lots and ret <= -stop_loss_pct:
                    close_lot(lot, prem, "stop_loss", j, ret)
            if time_exit:
                for lot in list(open_lots):
                    close_lot(lot, prem, time_exit, j, ret)
                break
            if not open_lots:
                break

    if open_lots:
        j, prem, ret, time_exit = steps[-1]
        for lot in list(open_lots):
            close_lot(lot, prem, time_exit or "eod", j, ret)

    total_pnl = float(sum(x["pnl"] for x in legs))
    reasons = "+".join(x["exit_reason"] for x in sorted(legs, key=lambda r: r["lot"]))
    return total_pnl, reasons, last_j, legs


def run_spy_day_backtest(
    bars: pd.DataFrame,
    *,
    cfg: SpyDayConfig | None = None,
    initial_equity: float = 100_000.0,
    amd_as_0dte: bool = False,
    qty: int = 1,
    scale_out: ScaleOutConfig | None = None,
    skip_fn: Any = None,
    tp_unlock_fn: Any = None,
) -> SpyDayBacktestResult:
    cfg = cfg or load_spy_day_config()
    configure_htf_permission(cfg.htf_permission)
    configure_ote_params(cfg.ote)
    patterns = list(cfg.patterns)
    if "amd" in patterns and cfg.amd.enabled:
        configure_amd_params(cfg.amd.detector_dict())
    elif "amd" in patterns:
        patterns = [p for p in patterns if p != "amd"]
    clear_ote_day_state()
    df = enrich_bars(bars, orb_minutes=cfg.orb_minutes)
    if df.empty:
        empty = pd.DataFrame()
        return SpyDayBacktestResult(
            trades=empty,
            equity_curve=pd.DataFrame([{"date": str(date.today()), "equity": initial_equity}]),
            metrics={
                "n_trades": 0,
                "expectancy": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
            },
            by_pattern={},
        )
    set_htf_cache(df)
    if "ote" in patterns:
        build_ote_cache(df)

    flat_parts = cfg.force_flat_et.split(":")
    force_flat = time(int(flat_parts[0]), int(flat_parts[1]) if len(flat_parts) > 1 else 0)
    amd_force_raw = cfg.amd.force_flat_et
    amd_no_tp1_raw = cfg.amd.flat_if_no_tp1_et
    amd_force_flat = bool(amd_force_raw)
    amd_flat_if_no_tp1 = (not amd_force_flat) and bool(amd_no_tp1_raw)
    amd_flat_clock = amd_force_raw or amd_no_tp1_raw
    if amd_flat_clock:
        amd_flat_parts = str(amd_flat_clock).split(":")
        amd_force_flat_time = time(
            int(amd_flat_parts[0]),
            int(amd_flat_parts[1]) if len(amd_flat_parts) > 1 else 0,
        )
    else:
        amd_force_flat_time = force_flat
    late_parts = cfg.no_new_entries_after_et.split(":")
    late = time(int(late_parts[0]), int(late_parts[1]) if len(late_parts) > 1 else 0)

    equity = initial_equity
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    trades_today = 0
    current_day = None
    exit_until_idx = -1

    closes = df["Close"]
    amd_prem_pct = 0.008

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
            enabled=patterns,
            min_confidence=cfg.score_threshold,
            pattern_min_confidence=cfg.pattern_min_confidence,
        )
        if sig is None:
            continue
        if skip_fn is not None and skip_fn(sig, i, df):
            continue

        spot = float(sig.spot)
        is_amd = sig.pattern == "amd" and not amd_as_0dte
        side = 1 if sig.side == "call" else -1
        unlock = tp_unlock_fn(ts, side, is_amd, df, i) if tp_unlock_fn is not None else None

        if is_amd:
            entry = float(np.clip(spot * amd_prem_pct, 0.80, 5.0))
            entry *= 1 + cfg.spread_penalty_pct / 2
            tp_pct = cfg.amd.profit_target_pct
            sl_pct = cfg.amd.stop_loss_pct
        else:
            entry = float(np.clip(spot * cfg.option_premium_pct_of_spot, 0.30, cfg.max_mid_price))
            entry *= 1 + cfg.spread_penalty_pct / 2
            tp_pct = cfg.profit_target_pct
            sl_pct = cfg.stop_loss_pct

        # Notional gate only (premium already clipped; spread bump may edge past mid)
        if entry * 100 > cfg.max_notional_per_trade:
            continue

        lots = int(scale_out.qty_for(sig.pattern)) if scale_out is not None else max(1, int(qty))
        if scale_out is not None:
            so = scale_out.for_pattern(sig.pattern)
            pnl, reason, exit_j, legs = simulate_scale_out(
                closes,
                ts,
                entry,
                spot,
                side,
                profit_target_pct=tp_pct,
                stop_loss_pct=sl_pct,
                short_dte=is_amd,
                force_flat=amd_force_flat_time if is_amd else force_flat,
                spread_penalty_pct=cfg.spread_penalty_pct,
                scale=so,
                tp_unlock_idx=unlock,
                amd_flat_if_no_tp1=is_amd and amd_flat_if_no_tp1,
                amd_force_flat=is_amd and amd_force_flat,
            )
            exit_prem = float(np.mean([lg["exit"] for lg in legs])) if legs else entry
        elif is_amd:
            exit_prem, reason, exit_j = _simulate_short_dte_path(
                closes,
                ts,
                entry,
                spot,
                side,
                profit_target_pct=tp_pct,
                stop_loss_pct=sl_pct,
                max_sessions=3,
                spread_penalty_pct=cfg.spread_penalty_pct,
            )
            pnl = (exit_prem - entry) * 100 * lots
            legs = []
        else:
            exit_prem, reason, exit_j = _simulate_0dte_path(
                closes,
                ts,
                entry,
                spot,
                side,
                profit_target_pct=tp_pct,
                stop_loss_pct=sl_pct,
                force_flat=force_flat,
                spread_penalty_pct=cfg.spread_penalty_pct,
                tp_unlock_idx=unlock,
            )
            pnl = (exit_prem - entry) * 100 * lots
            legs = []

        equity += pnl
        trades_today += 1
        exit_until_idx = exit_j if exit_j >= 0 else min(len(df) - 1, i + 1)

        trades.append(
            {
                "datetime": str(ts),
                "date": str(d),
                "pattern": sig.pattern,
                "lane": "amd_short_dte" if is_amd else "0dte",
                "side": sig.side,
                "confidence": sig.confidence,
                "reason": sig.reason,
                "spot": spot,
                "entry": entry,
                "exit": exit_prem,
                "qty": lots,
                "pnl": pnl,
                "exit_reason": reason,
                "leg_reasons": "+".join(lg["exit_reason"] for lg in legs) if legs else reason,
                "equity": equity,
            }
        )
        equity_rows.append({"datetime": str(ts), "equity": equity})

    trades_df = pd.DataFrame(trades)
    curve = pd.DataFrame(equity_rows)
    if curve.empty:
        curve = pd.DataFrame([{"datetime": str(datetime.now(ET)), "equity": initial_equity}])

    metrics = _metrics(trades_df, curve, initial_equity)
    by_pattern: dict[str, dict[str, float]] = {}
    if len(trades_df):
        for pat, g in trades_df.groupby("pattern"):
            by_pattern[str(pat)] = {
                "n_trades": float(len(g)),
                "win_rate": float((g["pnl"] > 0).mean()),
                "expectancy": float(g["pnl"].mean()),
                "profit_factor": _pf(g["pnl"]),
                "total_pnl": float(g["pnl"].sum()),
            }
    clear_htf_cache()
    clear_ote_cache()
    try:
        from stockpro.spy_day.intraday_patterns import clear_intraday_caches

        clear_intraday_caches()
    except Exception:
        pass
    return SpyDayBacktestResult(
        trades=trades_df, equity_curve=curve, metrics=metrics, by_pattern=by_pattern
    )


def _pf(pnls: pd.Series) -> float:
    wins = pnls[pnls > 0].sum()
    losses = pnls[pnls < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / abs(losses))


def _metrics(trades: pd.DataFrame, curve: pd.DataFrame, initial_equity: float) -> dict[str, float]:
    if trades is None or len(trades) == 0:
        return {
            "final_equity": float(initial_equity),
            "total_return": 0.0,
            "n_trades": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "trades_per_day": 0.0,
        }
    n_days = trades["date"].nunique() if "date" in trades.columns else 1
    return {
        "final_equity": float(curve["equity"].iloc[-1]),
        "total_return": float(curve["equity"].iloc[-1] / initial_equity - 1),
        "n_trades": float(len(trades)),
        "max_drawdown": _max_drawdown(curve["equity"]),
        "win_rate": float((trades["pnl"] > 0).mean()),
        "expectancy": float(trades["pnl"].mean()),
        "profit_factor": _pf(trades["pnl"]),
        "trades_per_day": float(len(trades) / max(n_days, 1)),
    }
