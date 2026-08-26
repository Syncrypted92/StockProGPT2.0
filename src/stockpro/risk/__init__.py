"""Risk management: sizing, daily/weekly loss limits, kill switch."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from stockpro.config import Settings

_OCC_RE = re.compile(r"^([A-Z]+)(\d{6})[CP]\d{8}$")


def underlying_from_option_symbol(symbol: str) -> str:
    """Extract root ticker from an OCC option symbol (e.g. INTC260731C...)."""
    sym = str(symbol or "").strip().upper()
    m = _OCC_RE.match(sym)
    if m:
        return m.group(1)
    # Synthetic / fallback: letters before first digit
    m2 = re.match(r"^([A-Z]+)", sym)
    return m2.group(1) if m2 else sym


def option_right(symbol: str) -> str:
    """Return 'call' or 'put' from OCC option symbol; empty if unknown."""
    m = re.search(r"\d{6}([CP])\d{8}$", str(symbol or "").upper())
    if not m:
        return ""
    return "call" if m.group(1) == "C" else "put"


def amd_swing_signal(
    *,
    right: str,
    ret: float,
    arm_pct: float = 0.35,
    arm_hour_et: float | None = None,
    or_extension: float | None = None,
    und_move: float | None = None,
    mom_30m: float | None = None,
    gate: str = "i1",
) -> tuple[bool, list[str]]:
    """Decide swing vs hard TP for an AMD book (paper gate).

    Evaluated when premium reaches arm_pct (+35% live).
    Default rule remains take TP; return True only to trail instead.

    Gates:
      i1 — AMD CALL at/above arm (tag I1_call). Paper default after I1 vs C5 BT.
      c5 — CALL + arm before 11:00 ET + OR ext >= 15 bps (tag C5_call_morning_ext).

    Puts always False -> hard TP.
    """
    tags: list[str] = []
    g = (gate or "i1").strip().lower()
    if right != "call":
        tags.append("put_hard_tp")
        return False, tags
    if ret < arm_pct:
        tags.append("call_below_arm")
        return False, tags
    tags.append("I1_call")

    if g in ("i1", "call", "amd_call_at_arm"):
        tags.append("I1_swing")
        return True, tags

    # C5 checklist
    ok_morning = arm_hour_et is not None and arm_hour_et < 11.0
    ok_ext = or_extension is not None and or_extension >= 0.0015
    if ok_morning:
        tags.append("I3_arm_before_11")
    else:
        tags.append("fail_morning")
    if ok_ext:
        tags.append("I7_or_ext_15bps")
    else:
        tags.append("fail_or_ext")
    if und_move is not None and und_move >= 0.010:
        tags.append("I5_strong_und_1pct")
    if mom_30m is not None and mom_30m > 0:
        tags.append("I9_mom30_pos")

    if ok_morning and ok_ext:
        tags.append("C5_call_morning_ext")
        return True, tags
    tags.append("call_at_arm_no_swing")
    return False, tags


def should_exit_amd_call_swing(
    entry_premium: float,
    current_premium: float,
    *,
    peak_ret: float,
    arm_pct: float = 0.35,
    trail_pct: float = 0.20,
    stop_loss_pct: float = 0.30,
    runner_cap_pct: float = 1.50,
) -> tuple[bool, str, float]:
    """After swing signal: no hard TP; trail off peak (SL + runner cap still apply).

    Returns (should_exit, reason, updated_peak_ret).
    """
    if entry_premium <= 0:
        return True, "invalid_entry", peak_ret
    if current_premium <= 0:
        return False, "", peak_ret
    ret = (current_premium - entry_premium) / entry_premium
    if ret <= -stop_loss_pct:
        return True, "stop_loss", peak_ret
    peak = max(float(peak_ret), ret)
    if ret >= runner_cap_pct:
        return True, "runner_cap", peak
    if peak >= arm_pct and ret <= (peak - trail_pct) + 1e-9:
        return True, "trail_stop", peak
    return False, "", peak


@dataclass
class ScaleOutState:
    original_qty: int
    tp1_done: bool = False
    tp2_done: bool = False
    be_done: bool = False
    peak_ret: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_qty": int(self.original_qty),
            "tp1_done": bool(self.tp1_done),
            "tp2_done": bool(self.tp2_done),
            "be_done": bool(self.be_done),
            "peak_ret": float(self.peak_ret),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None, *, qty: int) -> "ScaleOutState":
        raw = dict(raw or {})
        orig = int(raw.get("original_qty") or qty or 1)
        return cls(
            original_qty=max(1, orig),
            tp1_done=bool(raw.get("tp1_done", False)),
            tp2_done=bool(raw.get("tp2_done", False)),
            be_done=bool(raw.get("be_done", False)),
            peak_ret=float(raw.get("peak_ret") or 0.0),
        )


def next_scale_out_action(
    qty: int,
    ret: float,
    *,
    tp1_pct: float,
    tp2_pct: float,
    stop_loss_pct: float,
    state: ScaleOutState,
    time_flat: bool = False,
    runner_stop_at_entry: bool = True,
    runner_trail_pct: float | None = None,
) -> tuple[int, str, ScaleOutState]:
    """One scale-out action for the current book.

    Returns (close_qty, reason, new_state). close_qty=0 means hold.
    3-lot: bank 1 at TP1, 1 at tp2 (original SL), runner trails after TP2
    then SL→entry as floor.
    """
    qty = int(qty)
    st = ScaleOutState(
        original_qty=max(int(state.original_qty), qty, 1),
        tp1_done=bool(state.tp1_done),
        tp2_done=bool(state.tp2_done),
        be_done=bool(state.be_done),
        peak_ret=float(state.peak_ret or 0.0),
    )
    if qty <= 0:
        return 0, "", st

    if time_flat:
        return qty, "session_flat", st

    if not st.tp1_done:
        if ret <= -stop_loss_pct:
            return qty, "stop_loss", st
        if ret >= tp1_pct:
            st.tp1_done = True
            return min(1, qty), "profit_target", st
        return 0, "", st

    if ret >= tp2_pct and not st.tp2_done and qty >= 1:
        st.tp2_done = True
        st.peak_ret = max(st.peak_ret, ret)
        return min(1, qty), "profit_target_2", st

    if ret <= -stop_loss_pct:
        return qty, "stop_loss", st

    if st.tp2_done:
        st.peak_ret = max(st.peak_ret, ret)
        trail = float(runner_trail_pct or 0.0)
        if trail > 0 and (st.peak_ret - ret) >= trail - 1e-12:
            return qty, "runner_trail", st

    if runner_stop_at_entry and ret <= 0.0 and not st.be_done:
        if st.tp2_done:
            st.be_done = True
            return qty, "breakeven", st
        if qty >= 2:
            st.be_done = True
            return 1, "breakeven", st
        return 0, "", st

    return 0, "", st


def open_underlyings(positions: list[dict[str, Any]]) -> set[str]:
    """Unique underlyings currently held (option roots)."""
    out: set[str] = set()
    for p in positions:
        sym = str(p.get("symbol") or "")
        if len(sym) < 10:
            continue
        out.add(underlying_from_option_symbol(sym))
    return out


def stop_cooldown_tickers(
    trades: pd.DataFrame,
    *,
    cooldown_days: int,
    as_of: date | None = None,
) -> dict[str, date]:
    """Tickers still in post-stop cooldown → last stop date."""
    as_of = as_of or date.today()
    if cooldown_days <= 0 or trades is None or trades.empty:
        return {}
    df = trades.copy()
    if "exit_reason" not in df.columns or "timestamp" not in df.columns:
        return {}
    stops = df[df["exit_reason"].astype(str).str.lower() == "stop_loss"].copy()
    if stops.empty:
        return {}
    stops["ts"] = pd.to_datetime(stops["timestamp"], utc=True, errors="coerce")
    stops = stops.dropna(subset=["ts"])
    blocked: dict[str, date] = {}
    for _, row in stops.iterrows():
        ticker = str(row.get("ticker") or "")
        # Exit rows often store OCC symbol in ticker; normalize to root
        root = underlying_from_option_symbol(ticker) if len(ticker) > 6 else ticker.upper()
        stop_day = row["ts"].date()
        # cooldown_days trading days ≈ calendar via BDay
        until = (pd.Timestamp(stop_day) + pd.tseries.offsets.BDay(cooldown_days)).date()
        if as_of <= until:
            prev = blocked.get(root)
            if prev is None or stop_day > prev:
                blocked[root] = stop_day
    return blocked


def ticker_stop_rate(
    trades: pd.DataFrame,
    ticker: str,
    *,
    lookback: int = 5,
) -> float | None:
    """Fraction of last N closed exits for ticker that were stop_loss."""
    if trades is None or trades.empty or lookback <= 0:
        return None
    if "exit_reason" not in trades.columns:
        return None
    root = ticker.upper()
    df = trades.copy()
    df["_root"] = df["ticker"].astype(str).map(
        lambda t: underlying_from_option_symbol(t) if len(str(t)) > 6 else str(t).upper()
    )
    closed = df[df["exit_reason"].astype(str).str.len() > 0]
    closed = closed[closed["_root"] == root]
    if closed.empty:
        return None
    if "timestamp" in closed.columns:
        closed = closed.sort_values("timestamp")
    recent = closed.tail(lookback)
    if recent.empty:
        return None
    stops = (recent["exit_reason"].astype(str).str.lower() == "stop_loss").sum()
    return float(stops) / float(len(recent))


@dataclass
class RiskDecision:
    allowed: bool
    qty: int
    reasons: list[str] = field(default_factory=list)
    max_loss_dollars: float = 0.0


@dataclass
class RiskState:
    """Tracks realized PnL for daily/weekly halt logic."""

    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    day: date | None = None
    week_start: date | None = None
    halted_until_manual: bool = False
    open_positions: int = 0

    def roll_calendar(self, today: date | None = None) -> None:
        today = today or date.today()
        if self.day != today:
            self.day = today
            self.daily_pnl = 0.0
        week_start = today - timedelta(days=today.weekday())
        if self.week_start != week_start:
            self.week_start = week_start
            self.weekly_pnl = 0.0


class RiskManager:
    def __init__(self, settings: Settings, state: RiskState | None = None):
        self.settings = settings
        self.state = state or RiskState()
        self.state.roll_calendar()

    @property
    def cfg(self) -> dict[str, Any]:
        return self.settings.get("risk", default={}) or {}

    def is_halted(self) -> tuple[bool, str]:
        if self.settings.trading_halted or self.state.halted_until_manual:
            return True, "TRADING_HALTED"
        equity_proxy = 1.0  # ratios applied against tracked pnl vs equity at check time
        _ = equity_proxy
        return False, ""

    def check_loss_limits(self, equity: float) -> tuple[bool, list[str]]:
        """Return (ok_to_trade, reasons). May set weekly halt."""
        self.state.roll_calendar()
        reasons: list[str] = []
        if self.settings.trading_halted or self.state.halted_until_manual:
            reasons.append("trading halted")
            return False, reasons

        max_daily = float(self.cfg.get("max_daily_loss_pct", 0.02))
        max_weekly = float(self.cfg.get("max_weekly_loss_pct", 0.05))

        if equity > 0 and self.state.daily_pnl <= -max_daily * equity:
            reasons.append(
                f"daily loss {self.state.daily_pnl:.2f} exceeds {max_daily:.1%} of equity"
            )
        if equity > 0 and self.state.weekly_pnl <= -max_weekly * equity:
            self.state.halted_until_manual = True
            reasons.append(
                f"weekly loss {self.state.weekly_pnl:.2f} exceeds {max_weekly:.1%}; "
                "manual re-enable required"
            )
        return len(reasons) == 0, reasons

    def size_order(
        self,
        equity: float,
        premium: float,
        open_positions: int | None = None,
    ) -> RiskDecision:
        self.state.roll_calendar()
        reasons: list[str] = []
        open_pos = self.state.open_positions if open_positions is None else open_positions

        ok, loss_reasons = self.check_loss_limits(equity)
        reasons.extend(loss_reasons)
        if not ok:
            return RiskDecision(allowed=False, qty=0, reasons=reasons)

        max_open = int(self.cfg.get("max_open_positions", 3))
        if open_pos >= max_open:
            reasons.append(f"open positions {open_pos} >= max {max_open}")
            return RiskDecision(allowed=False, qty=0, reasons=reasons)

        if premium <= 0:
            reasons.append("invalid premium")
            return RiskDecision(allowed=False, qty=0, reasons=reasons)

        max_risk_pct = float(self.cfg.get("max_risk_per_trade_pct", 0.01))
        max_notional = float(self.cfg.get("max_notional_per_trade", 500.0))
        max_contracts = int(self.cfg.get("max_contracts_per_trade", 1))

        # Long option: max loss ~= premium * 100 * qty
        risk_budget = min(equity * max_risk_pct, max_notional)
        per_contract_cost = premium * 100
        qty = int(risk_budget // per_contract_cost)
        qty = max(0, min(qty, max_contracts))

        if qty < 1:
            reasons.append(
                f"premium ${per_contract_cost:.2f} exceeds risk budget ${risk_budget:.2f}"
            )
            return RiskDecision(allowed=False, qty=0, reasons=reasons)

        return RiskDecision(
            allowed=True,
            qty=qty,
            reasons=["sized ok"],
            max_loss_dollars=per_contract_cost * qty,
        )

    def record_realized_pnl(self, pnl: float) -> None:
        self.state.roll_calendar()
        self.state.daily_pnl += pnl
        self.state.weekly_pnl += pnl

    def should_exit_long(
        self,
        entry_premium: float,
        current_premium: float,
        expiration: date,
        signal_flipped: bool,
        today: date | None = None,
        *,
        profit_target_pct: float | None = None,
        stop_loss_pct: float | None = None,
        exit_days_before_expiry: int | None = None,
        force_flat: bool = False,
    ) -> tuple[bool, str]:
        today = today or date.today()
        if force_flat:
            return True, "session_flat"
        if entry_premium <= 0:
            return True, "invalid_entry"
        ret = (current_premium - entry_premium) / entry_premium
        profit_tgt = float(
            self.cfg.get("profit_target_pct", 0.50)
            if profit_target_pct is None
            else profit_target_pct
        )
        stop = float(self.cfg.get("stop_loss_pct", 0.40) if stop_loss_pct is None else stop_loss_pct)
        days_before = int(
            self.cfg.get("exit_days_before_expiry", 7)
            if exit_days_before_expiry is None
            else exit_days_before_expiry
        )

        if ret >= profit_tgt:
            return True, "profit_target"
        if ret <= -stop:
            return True, "stop_loss"
        dte = (expiration - today).days
        # days_before < 0 disables the time stop (used for intraday 0DTE holds)
        if days_before >= 0 and dte <= days_before:
            return True, "time_stop"
        if signal_flipped:
            return True, "signal_flip"
        return False, ""


def load_pnl_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "pnl"])
    return pd.read_csv(path)
