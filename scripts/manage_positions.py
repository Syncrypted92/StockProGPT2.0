"""Manage open option positions: profit target, stop, time stop (incl. 0DTE rules)."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

from stockpro.broker import AlpacaBroker
from stockpro.config import ROOT, load_settings
from stockpro.journal import Journal
from stockpro.risk import (
    RiskManager,
    ScaleOutState,
    amd_swing_signal,
    next_scale_out_action,
    option_right,
    should_exit_amd_call_swing,
    underlying_from_option_symbol,
)

ET = ZoneInfo("America/New_York")
SWING_STATE_PATH = ROOT / "data" / "journal" / "amd_swing_state.json"
SCALE_STATE_PATH = ROOT / "data" / "journal" / "scale_out_state.json"


def parse_expiration(symbol: str) -> date | None:
    m = re.search(r"(\d{6})[CP]\d{8}$", symbol)
    if not m:
        return None
    yy, mm, dd = m.group(1)[:2], m.group(1)[2:4], m.group(1)[4:6]
    year = 2000 + int(yy)
    try:
        return date(year, int(mm), int(dd))
    except ValueError:
        return None


def _load_swing_state() -> dict:
    if not SWING_STATE_PATH.exists():
        return {}
    try:
        return json.loads(SWING_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save_swing_state(state: dict) -> None:
    SWING_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SWING_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _load_scale_state() -> dict:
    if not SCALE_STATE_PATH.exists():
        return {}
    try:
        return json.loads(SCALE_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save_scale_state(state: dict) -> None:
    SCALE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCALE_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _spy_swing_context(right: str, orb_minutes: int = 30) -> dict:
    """Live indicators for swing signal: OR extension, 30m mom, hour ET."""
    out: dict = {
        "arm_hour_et": None,
        "or_extension": None,
        "und_move": None,
        "mom_30m": None,
        "spot": None,
    }
    try:
        from stockpro.data import get_spy_5m

        bars = get_spy_5m(lookback_days=5)
        if bars is None or bars.empty:
            return out
        if bars.index.tz is None:
            bars.index = bars.index.tz_localize(ET)
        else:
            bars = bars.copy()
            bars.index = bars.index.tz_convert(ET)
        now = datetime.now(ET)
        out["arm_hour_et"] = now.hour + now.minute / 60.0
        day = bars[bars.index.date == now.date()]
        if day.empty:
            return out
        n_orb = max(1, orb_minutes // 5)
        orb = day.iloc[:n_orb]
        orb_hi = float(orb["High"].max())
        orb_lo = float(orb["Low"].min())
        spot = float(day["Close"].iloc[-1])
        out["spot"] = spot
        side = 1 if right == "call" else -1
        if side == 1:
            out["or_extension"] = (spot - orb_hi) / spot if spot else None
        else:
            out["or_extension"] = (orb_lo - spot) / spot if spot else None
        if len(day) >= 7:
            prev = float(day["Close"].iloc[-7])
            out["mom_30m"] = ((spot / prev) - 1.0) * side if prev > 0 else None
        # und_move vs day open as proxy if entry spot unknown
        day_open = float(day["Open"].iloc[0])
        if day_open > 0:
            out["und_move"] = ((spot / day_open) - 1.0) * side
    except Exception:  # noqa: BLE001
        pass
    return out


def run_manage(dry_run: bool = True) -> dict:
    settings = load_settings()
    if settings.trading_halted:
        raise RuntimeError("TRADING_HALTED=true — refusing to manage positions")

    zd = settings.get("zerodte", default={}) or {}
    spy_day = settings.get("spy_day", default={}) or {}
    spy_day_enabled = bool(spy_day.get("enabled", False))
    zd_enabled = bool(zd.get("enabled", False)) or spy_day_enabled
    amd_raw = spy_day.get("amd") if isinstance(spy_day.get("amd"), dict) else {}
    amd_patterns = list(spy_day.get("patterns") or [])
    amd_lane_on = bool(amd_raw.get("enabled", False)) and "amd" in amd_patterns
    amd_min_dte = int(amd_raw.get("min_dte", 2))
    amd_max_dte = int(amd_raw.get("max_dte", 5))
    amd_tp = float(amd_raw.get("profit_target_pct", 0.35))
    amd_sl = float(amd_raw.get("stop_loss_pct", 0.30))
    amd_swing_calls = bool(amd_raw.get("swing_calls_after_tp", False))
    amd_swing_gate = str(amd_raw.get("swing_gate", "i1")).strip().lower() or "i1"
    amd_swing_arm = float(amd_raw.get("swing_arm_pct", amd_tp))
    amd_swing_trail = float(amd_raw.get("swing_trail_pct", 0.20))
    amd_swing_cap = float(amd_raw.get("swing_runner_cap_pct", 1.50))
    scale_raw = spy_day.get("scale_out") if isinstance(spy_day.get("scale_out"), dict) else {}
    scale_on = spy_day_enabled and bool(scale_raw.get("enabled", False))
    scale_tp2 = float(scale_raw.get("tp2_pct", 0.60))
    scale_be = bool(scale_raw.get("runner_stop_at_entry", True))
    trail_raw = scale_raw.get("runner_trail_pct", 0.12)
    scale_trail = None if trail_raw in (None, "", False) else float(trail_raw)
    if scale_trail is not None and scale_trail <= 0:
        scale_trail = None
    # Prefer spy_day TP/SL/flat when that lane is on (0DTE ORB family)
    if spy_day_enabled:
        zd_tp = float(spy_day.get("profit_target_pct", 0.25))
        zd_sl = float(spy_day.get("stop_loss_pct", 0.30))
        zd_flat_et = str(spy_day.get("force_flat_et", "15:45"))
    else:
        zd_tp = float(zd.get("profit_target_pct", 0.25))
        zd_sl = float(zd.get("stop_loss_pct", 0.30))
        zd_flat_et = str(zd.get("force_flat_et", "15:45"))

    broker = AlpacaBroker(settings, dry_run=dry_run)
    broker.connect()
    risk = RiskManager(settings)
    journal = Journal(settings)
    positions = broker.list_positions()
    account = broker.get_account()
    now_et = datetime.now(ET)
    flat_h, flat_m = [int(x) for x in zd_flat_et.split(":")[:2]]
    past_0dte_flat = (now_et.hour, now_et.minute) >= (flat_h, flat_m)

    swing_state = _load_swing_state()
    scale_state = _load_scale_state()
    open_syms = {str(p["symbol"]) for p in positions if len(str(p.get("symbol") or "")) >= 15}
    # Drop state for closed symbols
    swing_state = {k: v for k, v in swing_state.items() if k in open_syms}
    scale_state = {k: v for k, v in scale_state.items() if k in open_syms}

    actions = []
    for pos in positions:
        symbol = pos["symbol"]
        if len(symbol) < 15:
            continue

        entry = float(pos["avg_entry_price"])
        current = float(pos.get("current_price") or 0)
        if current <= 0:
            quotes = broker.get_option_quotes([symbol])
            q = quotes.get(symbol) or {}
            bid, ask = q.get("bid", 0.0), q.get("ask", 0.0)
            if bid > 0 and ask > 0:
                current = (bid + ask) / 2
            elif bid > 0:
                current = bid

        expiration = parse_expiration(symbol) or date.today()
        dte = (expiration - date.today()).days
        root = underlying_from_option_symbol(symbol)
        right = option_right(symbol)
        is_0dte = dte == 0
        # Short-DTE AMD paper lane: same SPY, not same-day 0DTE — separate TP/SL, no EOD forced flat.
        # Include dte==1 so an AMD book aging into 1 DTE keeps AMD exits (not 0DTE session flat).
        is_amd_lane = (
            amd_lane_on
            and root == str(spy_day.get("symbol", "SPY")).upper()
            and (amd_min_dte <= dte <= amd_max_dte or dte == 1)
        )
        prem_now = current if current > 0 else entry
        ret = (prem_now - entry) / entry if entry > 0 else 0.0
        qty = abs(int(float(pos.get("qty") or 0)))
        use_scale = scale_on and qty >= 1 and (is_amd_lane or (is_0dte and zd_enabled))
        swing_on, swing_tags = (False, [])
        ctx: dict = {}
        prev_st = swing_state.get(symbol) or {}
        # Scale-out replaces AMD I1 trail (matches backtest).
        if (not use_scale) and is_amd_lane and amd_swing_calls and prev_st.get("armed") and prev_st.get("swing_signal"):
            swing_on, swing_tags = True, list(prev_st.get("signal_tags") or ["C5_latched"])
        elif (not use_scale) and is_amd_lane and amd_swing_calls:
            ctx = _spy_swing_context(right, orb_minutes=int(spy_day.get("orb_minutes", 30)))
            # If already above arm, use stored arm hour if we have it
            arm_hour = prev_st.get("arm_hour_et", ctx.get("arm_hour_et"))
            swing_on, swing_tags = amd_swing_signal(
                right=right,
                ret=ret,
                arm_pct=amd_swing_arm,
                arm_hour_et=arm_hour,
                or_extension=ctx.get("or_extension"),
                und_move=ctx.get("und_move"),
                mom_30m=ctx.get("mom_30m"),
                gate=amd_swing_gate,
            )
        swing_meta: dict = {}
        close_qty = 0
        should = False
        reason = ""
        lane = "amd" if is_amd_lane else ("0dte" if is_0dte else "swing")

        if use_scale:
            tp1 = amd_tp if is_amd_lane else zd_tp
            sl = amd_sl if is_amd_lane else zd_sl
            time_flat = bool(is_0dte and (not is_amd_lane) and past_0dte_flat)
            if is_amd_lane and dte <= 0:
                time_flat = True
            st = ScaleOutState.from_dict(scale_state.get(symbol), qty=qty)
            if qty <= st.original_qty - 1:
                st.tp1_done = True
            if qty <= st.original_qty - 2:
                st.tp2_done = True
            close_qty, reason, st = next_scale_out_action(
                qty,
                ret,
                tp1_pct=tp1,
                tp2_pct=scale_tp2,
                stop_loss_pct=sl,
                state=st,
                time_flat=time_flat,
                runner_stop_at_entry=scale_be,
                runner_trail_pct=scale_trail,
            )
            should = close_qty > 0
            swing_meta = {
                "exit_mode": "scale_out",
                "ret": round(ret, 4),
                "qty": qty,
                "close_qty": close_qty,
                **st.to_dict(),
            }
            scale_state[symbol] = {
                **st.to_dict(),
                "entry": entry,
                "updated": now_et.isoformat(),
            }
        elif is_amd_lane and swing_on:
            prev_peak = float((swing_state.get(symbol) or {}).get("peak_ret", 0.0))
            should, reason, new_peak = should_exit_amd_call_swing(
                entry_premium=entry,
                current_premium=prem_now,
                peak_ret=prev_peak,
                arm_pct=amd_swing_arm,
                trail_pct=amd_swing_trail,
                stop_loss_pct=amd_sl,
                runner_cap_pct=amd_swing_cap,
            )
            if not should and dte <= 0:
                should, reason = True, "time_stop"
            was_armed = bool((swing_state.get(symbol) or {}).get("armed"))
            swing_state[symbol] = {
                "peak_ret": new_peak,
                "armed": True,
                "swing_signal": True,
                "signal_tags": swing_tags,
                "arm_hour_et": prev_st.get("arm_hour_et", ctx.get("arm_hour_et")),
                "updated": now_et.isoformat(),
                "entry": entry,
                "current": current,
            }
            swing_meta.update(
                {
                    "exit_mode": "swing_trail",
                    "peak_ret": round(new_peak, 4),
                    "armed": True,
                }
            )
            if not was_armed:
                journal.log_decision(
                    ticker=root,
                    action="swing_signal",
                    reason="amd_swing_vs_tp",
                    signal="swing",
                    contract=symbol,
                    entry=entry,
                    current=current,
                    details=json.dumps(
                        {"tags": swing_tags, "ret": round(ret, 4), "mode": "trail_after_arm"}
                    ),
                )
            if should:
                close_qty = qty
        elif is_amd_lane:
            should, reason = risk.should_exit_long(
                entry_premium=entry,
                current_premium=prem_now,
                expiration=expiration,
                signal_flipped=False,
                profit_target_pct=amd_tp,
                stop_loss_pct=amd_sl,
                exit_days_before_expiry=0,
                force_flat=False,
            )
            swing_meta["exit_mode"] = "hard_tp"
            if symbol in swing_state:
                del swing_state[symbol]
            if should:
                close_qty = qty
        elif is_0dte and zd_enabled:
            should, reason = risk.should_exit_long(
                entry_premium=entry,
                current_premium=current if current > 0 else entry,
                expiration=expiration,
                signal_flipped=False,
                profit_target_pct=zd_tp,
                stop_loss_pct=zd_sl,
                exit_days_before_expiry=-1 if dte == 0 else 0,
                force_flat=past_0dte_flat,
            )
            if reason == "time_stop" and dte == 0 and not past_0dte_flat:
                should, reason = risk.should_exit_long(
                    entry_premium=entry,
                    current_premium=current if current > 0 else entry,
                    expiration=expiration,
                    signal_flipped=False,
                    profit_target_pct=zd_tp,
                    stop_loss_pct=zd_sl,
                    exit_days_before_expiry=-1,
                    force_flat=False,
                )
            if should:
                close_qty = qty
        else:
            should, reason = risk.should_exit_long(
                entry_premium=entry,
                current_premium=current if current > 0 else entry,
                expiration=expiration,
                signal_flipped=False,
            )
            if should:
                close_qty = qty

        ok_day, _day_reasons = risk.check_loss_limits(float(account["equity"]))
        if not ok_day:
            should, reason = True, "daily_loss_halt"
            close_qty = qty

        if not should or close_qty <= 0:
            actions.append(
                {
                    "symbol": symbol,
                    "action": "hold",
                    "lane": lane,
                    "dte": dte,
                    "entry": entry,
                    "current": current,
                    "unrealized_pl": pos.get("unrealized_pl"),
                    **swing_meta,
                }
            )
            continue

        close_qty = min(close_qty, qty)
        result = broker.close_position(symbol, qty=close_qty)
        remaining = qty - close_qty
        unreal = float(pos.get("unrealized_pl") or 0.0)
        pnl = unreal * (close_qty / qty) if qty else unreal
        if result.submitted or result.dry_run:
            risk.record_realized_pnl(pnl)
        if remaining <= 0:
            swing_state.pop(symbol, None)
            scale_state.pop(symbol, None)
        journal.log_trade(
            ticker=root,
            side="sell_to_close",
            contract=symbol,
            qty=close_qty,
            limit_price=current,
            status=result.status,
            dry_run=result.dry_run,
            exit_reason=reason,
            pnl=pnl,
            lane=lane,
        )
        journal.log_decision(
            ticker=root,
            action="exit",
            reason=reason,
            entry=entry,
            current=current,
            dry_run=result.dry_run,
            status=result.status,
            contract=symbol,
            qty=close_qty,
            lane=lane,
            details=json.dumps(swing_meta) if swing_meta else "",
        )
        actions.append(
            {
                "symbol": symbol,
                "action": "exit",
                "reason": reason,
                "lane": lane,
                "dte": dte,
                "entry": entry,
                "current": current,
                "qty": close_qty,
                "remaining": remaining,
                "status": result.status,
                "dry_run": result.dry_run,
                **swing_meta,
            }
        )

    if not dry_run:
        _save_swing_state(swing_state)
        _save_scale_state(scale_state)
    else:
        # Still persist peak tracking on dry-run so paper tests accumulate state when scanning
        _save_swing_state(swing_state)
        _save_scale_state(scale_state)

    journal.log_daily_pnl(
        day=date.today().isoformat(),
        pnl=risk.state.daily_pnl,
        equity=float(account["equity"]),
    )
    return {
        "paper": settings.paper,
        "dry_run": dry_run,
        "equity": account["equity"],
        "n_positions": len(positions),
        "now_et": now_et.isoformat(),
        "zerodte_enabled": zd_enabled,
        "spy_day_enabled": spy_day_enabled,
        "amd_lane_on": amd_lane_on,
        "amd_swing_calls": amd_swing_calls and (not scale_on),
        "amd_swing_gate": amd_swing_gate,
        "scale_out": scale_on,
        "runner_trail_pct": scale_trail,
        "actions": actions,
        "timestamp": datetime.now().isoformat() + "Z",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage / exit open option positions")
    parser.add_argument("--submit", action="store_true", help="Actually submit close orders")
    parser.add_argument("--dry-run", action="store_true", help="Force dry-run")
    args = parser.parse_args()
    dry = True
    if args.submit:
        dry = False
    if args.dry_run:
        dry = True
    print(json.dumps(run_manage(dry_run=dry), indent=2, default=str))


if __name__ == "__main__":
    main()
