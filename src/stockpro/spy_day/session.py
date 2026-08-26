"""Session helpers for SPY day lane (0DTE ORB + optional short-DTE AMD)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from stockpro.config import Settings, load_settings
from stockpro.spy_day.htf_permission import HtfPermissionConfig

ET = ZoneInfo("America/New_York")


@dataclass
class AmdLaneConfig:
    """Short-dated AMD paper lane (separate from 0DTE ORB economics)."""

    enabled: bool = True
    require_confirm: bool = False
    apply_htf: bool = True
    window_start: str = "10:00"
    window_end: str = "12:00"
    pierce_atr: float = 0.15
    skip_friday: bool = False
    one_per_day: bool = True
    min_body_frac: float = 0.35
    min_dte: int = 2
    max_dte: int = 5
    target_dte: int = 3
    profit_target_pct: float = 0.35
    stop_loss_pct: float = 0.30
    min_confidence: float = 0.74
    # Paper test: after call hits arm_pct, trail instead of hard TP (puts keep hard TP)
    swing_calls_after_tp: bool = True
    swing_gate: str = "i1"  # i1 = call@arm | c5 = call+morning+OR ext
    swing_arm_pct: float = 0.35
    swing_trail_pct: float = 0.20
    swing_runner_cap_pct: float = 1.50

    def detector_dict(self) -> dict[str, Any]:
        return {
            "require_confirm": self.require_confirm,
            "apply_htf": self.apply_htf,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "pierce_atr": self.pierce_atr,
            "skip_friday": self.skip_friday,
            "one_per_day": self.one_per_day,
            "min_body_frac": self.min_body_frac,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "AmdLaneConfig":
        raw = dict(raw or {})
        return cls(
            enabled=bool(raw.get("enabled", True)),
            require_confirm=bool(raw.get("require_confirm", False)),
            apply_htf=bool(raw.get("apply_htf", True)),
            window_start=str(raw.get("window_start", "10:00")),
            window_end=str(raw.get("window_end", "12:00")),
            pierce_atr=float(raw.get("pierce_atr", 0.15)),
            skip_friday=bool(raw.get("skip_friday", False)),
            one_per_day=bool(raw.get("one_per_day", True)),
            min_body_frac=float(raw.get("min_body_frac", 0.35)),
            min_dte=int(raw.get("min_dte", 2)),
            max_dte=int(raw.get("max_dte", 5)),
            target_dte=int(raw.get("target_dte", 3)),
            profit_target_pct=float(raw.get("profit_target_pct", 0.35)),
            stop_loss_pct=float(raw.get("stop_loss_pct", 0.30)),
            min_confidence=float(raw.get("min_confidence", 0.74)),
            swing_calls_after_tp=bool(raw.get("swing_calls_after_tp", True)),
            swing_gate=str(raw.get("swing_gate", "i1")).strip().lower() or "i1",
            swing_arm_pct=float(raw.get("swing_arm_pct", raw.get("profit_target_pct", 0.35))),
            swing_trail_pct=float(raw.get("swing_trail_pct", 0.20)),
            swing_runner_cap_pct=float(raw.get("swing_runner_cap_pct", 1.50)),
        )


@dataclass
class ScaleOutConfig:
    """Live paper scale-out: bank 1 at TP1, remaining to tp2 / runner.

    qty_by_pattern overrides qty (e.g. orb=2, amd=3, power_hour=3).
    0DTE still force-flats 15:45 — no overnight swing except AMD short-DTE.
    """

    enabled: bool = False
    qty: int = 3
    tp2_pct: float = 0.60
    runner_trail_pct: float | None = 0.12
    runner_stop_at_entry: bool = True
    qty_by_pattern: dict[str, int] = field(default_factory=dict)

    def qty_for(self, pattern: str) -> int:
        if pattern in self.qty_by_pattern:
            return max(1, int(self.qty_by_pattern[pattern]))
        return max(1, int(self.qty))

    def for_pattern(self, pattern: str) -> "ScaleOutConfig":
        return ScaleOutConfig(
            enabled=self.enabled,
            qty=self.qty_for(pattern),
            tp2_pct=self.tp2_pct,
            runner_trail_pct=self.runner_trail_pct,
            runner_stop_at_entry=self.runner_stop_at_entry,
            qty_by_pattern=dict(self.qty_by_pattern),
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ScaleOutConfig":
        raw = dict(raw or {})
        qbp_raw = raw.get("qty_by_pattern") or {}
        qbp = {str(k): int(v) for k, v in dict(qbp_raw).items()} if isinstance(qbp_raw, dict) else {}
        trail_raw = raw.get("runner_trail_pct", 0.12)
        trail = None if trail_raw in (None, "", False) else float(trail_raw)
        if trail is not None and trail <= 0:
            trail = None
        return cls(
            enabled=bool(raw.get("enabled", False)),
            qty=int(raw.get("qty", 3)),
            tp2_pct=float(raw.get("tp2_pct", 0.60)),
            runner_trail_pct=trail,
            runner_stop_at_entry=bool(raw.get("runner_stop_at_entry", True)),
            qty_by_pattern=qbp,
        )


@dataclass
class SpyDayConfig:
    enabled: bool = True
    symbol: str = "SPY"
    history_days: int = 180
    patterns: list[str] = field(default_factory=lambda: ["orb"])
    orb_minutes: int = 30
    score_threshold: float = 0.65
    max_trades_per_day: int = 3
    max_open_positions: int = 2
    no_new_entries_after_et: str = "14:30"
    force_flat_et: str = "15:45"
    min_dte: int = 0
    max_dte: int = 0
    max_mid_price: float = 3.0
    min_mid_price: float = 0.40
    max_notional_per_trade: float = 1500.0
    max_contracts_per_trade: int = 3
    profit_target_pct: float = 0.30
    stop_loss_pct: float = 0.25
    # Slight ITM to modest OTM (~ATM). 1.5% OTM was letting in lottery 0DTE.
    otm_pct_min: float = -0.003
    otm_pct_max: float = 0.006
    target_delta_min: float = 0.35
    target_delta_max: float = 0.55
    target_delta: float = 0.42
    min_open_interest: int = 50
    min_option_volume: int = 10
    option_premium_pct_of_spot: float = 0.002
    spread_penalty_pct: float = 0.04
    backtest_gate_pf: float = 1.2
    backtest_gate_min_trades: int = 40
    pattern_min_confidence: dict[str, float] = field(default_factory=dict)
    htf_permission: HtfPermissionConfig = field(default_factory=HtfPermissionConfig)
    ote: dict[str, Any] = field(default_factory=dict)
    amd: AmdLaneConfig = field(default_factory=AmdLaneConfig)
    scale_out: ScaleOutConfig = field(default_factory=ScaleOutConfig)


def load_spy_day_config(settings: Settings | None = None) -> SpyDayConfig:
    settings = settings or load_settings()
    raw: dict[str, Any] = dict(settings.get("spy_day", default={}) or {})
    pmc_raw = raw.get("pattern_min_confidence") or {}
    pmc = {str(k): float(v) for k, v in dict(pmc_raw).items()} if isinstance(pmc_raw, dict) else {}
    htf_raw = raw.get("htf_permission") if isinstance(raw.get("htf_permission"), dict) else {}
    ote_raw = dict(raw.get("ote") or {}) if isinstance(raw.get("ote"), dict) else {}
    amd_raw = dict(raw.get("amd") or {}) if isinstance(raw.get("amd"), dict) else {}
    amd = AmdLaneConfig.from_dict(amd_raw)
    scale_raw = dict(raw.get("scale_out") or {}) if isinstance(raw.get("scale_out"), dict) else {}
    scale_out = ScaleOutConfig.from_dict(scale_raw)
    if "amd" not in pmc:
        pmc["amd"] = amd.min_confidence
    return SpyDayConfig(
        enabled=bool(raw.get("enabled", True)),
        symbol=str(raw.get("symbol", "SPY")),
        history_days=int(raw.get("history_days", 180)),
        patterns=list(raw.get("patterns") or ["orb"]),
        orb_minutes=int(raw.get("orb_minutes", 30)),
        score_threshold=float(raw.get("score_threshold", 0.65)),
        max_trades_per_day=int(raw.get("max_trades_per_day", 3)),
        max_open_positions=int(raw.get("max_open_positions", 2)),
        no_new_entries_after_et=str(raw.get("no_new_entries_after_et", "14:30")),
        force_flat_et=str(raw.get("force_flat_et", "15:45")),
        min_dte=int(raw.get("min_dte", 0)),
        max_dte=int(raw.get("max_dte", 0)),
        max_mid_price=float(raw.get("max_mid_price", 3.0)),
        min_mid_price=float(raw.get("min_mid_price", 0.40)),
        max_notional_per_trade=float(raw.get("max_notional_per_trade", 1500.0)),
        max_contracts_per_trade=int(raw.get("max_contracts_per_trade", 3)),
        profit_target_pct=float(raw.get("profit_target_pct", 0.30)),
        stop_loss_pct=float(raw.get("stop_loss_pct", 0.25)),
        otm_pct_min=float(raw.get("otm_pct_min", -0.003)),
        otm_pct_max=float(raw.get("otm_pct_max", 0.006)),
        target_delta_min=float(raw.get("target_delta_min", 0.35)),
        target_delta_max=float(raw.get("target_delta_max", 0.55)),
        target_delta=float(raw.get("target_delta", 0.42)),
        min_open_interest=int(raw.get("min_open_interest", 50)),
        min_option_volume=int(raw.get("min_option_volume", 10)),
        option_premium_pct_of_spot=float(raw.get("option_premium_pct_of_spot", 0.002)),
        spread_penalty_pct=float(raw.get("spread_penalty_pct", 0.04)),
        backtest_gate_pf=float(raw.get("backtest_gate_pf", 1.2)),
        backtest_gate_min_trades=int(raw.get("backtest_gate_min_trades", 40)),
        pattern_min_confidence=pmc,
        htf_permission=HtfPermissionConfig.from_dict(htf_raw),
        ote=ote_raw,
        amd=amd,
        scale_out=scale_out,
    )


def _parse_hhmm(s: str) -> time:
    parts = s.split(":")
    return time(int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)


def session_allows_entry(
    now: datetime | None = None,
    *,
    cfg: SpyDayConfig | None = None,
) -> tuple[bool, str]:
    """Entry window: after ORB forms (~10:00) until no_new_entries_after_et."""
    cfg = cfg or load_spy_day_config()
    now = now or datetime.now(ET)
    if now.tzinfo is None:
        now = now.replace(tzinfo=ET)
    else:
        now = now.astimezone(ET)
    if now.weekday() >= 5:
        return False, "weekend"
    t = now.time()
    if t < time(9, 35):
        return False, "pre_orb"
    cutoff = _parse_hhmm(cfg.no_new_entries_after_et)
    if t >= cutoff:
        return False, "too_late"
    flat = _parse_hhmm(cfg.force_flat_et)
    if t >= flat:
        return False, "force_flat_window"
    return True, "ok"


def past_force_flat(now: datetime | None = None, *, cfg: SpyDayConfig | None = None) -> bool:
    cfg = cfg or load_spy_day_config()
    now = now or datetime.now(ET)
    if now.tzinfo is None:
        now = now.replace(tzinfo=ET)
    else:
        now = now.astimezone(ET)
    flat = _parse_hhmm(cfg.force_flat_et)
    return now.time() >= flat
