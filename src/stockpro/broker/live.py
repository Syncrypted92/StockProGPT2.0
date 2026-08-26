"""Helpers for enabling live trading safely."""

from __future__ import annotations

from stockpro.config import Settings


def assert_live_allowed(settings: Settings) -> None:
    """Raise if live trading prerequisites are not met."""
    if settings.paper:
        raise RuntimeError("Live trading requested but PAPER=true")
    if not settings.allow_live:
        raise RuntimeError("Live trading blocked: set ALLOW_LIVE=true")
    if settings.trading_halted:
        raise RuntimeError("Live trading blocked: TRADING_HALTED=true")
    settings.require_broker_credentials()


def live_risk_overrides() -> dict:
    """Recommended tiny-size overrides when flipping to live."""
    return {
        "max_contracts_per_trade": 1,
        "max_notional_per_trade": 250.0,
        "max_open_positions": 2,
        "max_daily_loss_pct": 0.01,
        "max_weekly_loss_pct": 0.03,
        "dry_run": False,
    }
