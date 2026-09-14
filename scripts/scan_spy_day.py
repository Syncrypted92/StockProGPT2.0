"""SPY day paper scan: 0DTE ORB family + short-DTE AMD."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from stockpro.broker import AlpacaBroker, OrderRequest
from stockpro.config import ROOT, load_settings
from stockpro.data import get_spy_5m
from stockpro.journal import Journal
from stockpro.options import (
    contracts_from_alpaca_payload,
    select_contract,
    synthetic_candidates_for_backtest,
)
from stockpro.risk import RiskManager, classify_spy_option_books, underlying_from_option_symbol
from stockpro.spy_day.session import load_spy_day_config, session_allows_entry

ET = ZoneInfo("America/New_York")

# Two 5m bars of slack; beyond this the cache is not being refreshed.
STALE_BARS_MIN = 12.0


def _option_dte(symbol: str, today: date | None = None) -> int | None:
    """OCC-style expiry YYMMDD embedded in option symbol → calendar DTE."""
    from stockpro.risk import option_dte

    return option_dte(symbol, today)


def _enrich_quotes(broker: AlpacaBroker, candidates: list) -> None:
    symbols = [c.symbol for c in candidates if c.symbol and "_SYN_" not in c.symbol]
    if not symbols:
        return
    quotes = broker.get_option_quotes(symbols[:200])
    for c in candidates:
        q = quotes.get(c.symbol)
        if not q:
            continue
        bid, ask = q.get("bid", 0.0), q.get("ask", 0.0)
        if bid > 0 and ask > 0 and ask >= bid:
            c.bid = bid
            c.ask = ask
            c.mid = (bid + ask) / 2
            c.spread_pct = (ask - bid) / c.mid if c.mid > 0 else 1.0


def _gate_ok() -> tuple[bool, str]:
    path = ROOT / "artifacts" / "spy_day_backtest_latest.json"
    if not path.exists():
        return False, "no_backtest_artifact"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"bad_backtest_artifact:{exc}"
    if not data.get("gate_ok"):
        return False, "backtest_gate_fail"
    return True, "ok"


def run_spy_day_scan(*, dry_run: bool = True, submit: bool = False, refresh: bool = False) -> dict:
    settings = load_settings()
    cfg = load_spy_day_config(settings)
    if not cfg.enabled:
        return {"action": "disabled", "reason": "spy_day.enabled=false"}

    if settings.trading_halted:
        raise RuntimeError("TRADING_HALTED=true")

    if submit:
        dry_run = False

    # Refresh before session gate so early/skip runs still keep the cache warm.
    if refresh:
        get_spy_5m(settings, refresh=True, rth_only=True)

    ok_entry, why = session_allows_entry(cfg=cfg)
    if not ok_entry:
        return {
            "action": "skip_session",
            "reason": why,
            "now_et": datetime.now(ET).isoformat(),
            "dry_run": dry_run,
        }

    gate_ok, gate_reason = _gate_ok()
    # Allow dry-run even if gate fails; block live submit
    if submit and not gate_ok:
        return {
            "action": "blocked_gate",
            "reason": gate_reason,
            "hint": "Run scripts/backtest_spy_day.py until gate_ok=true",
            "dry_run": True,
        }

    from stockpro.spy_day.amd import configure_amd_params
    from stockpro.spy_day.mtf_liquidity import clear_htf_cache, set_htf_cache
    from stockpro.spy_day.patterns import (
        best_signal,
        clear_ote_day_state,
        configure_htf_permission,
        configure_ote_params,
        enrich_bars,
    )

    bars = get_spy_5m(settings, refresh=False, rth_only=True)
    if bars.empty or len(bars) < 50:
        return {"action": "skip_no_bars", "n_bars": len(bars)}

    last_ts = bars.index[-1]
    bar_age_min = None
    if last_ts.tzinfo is not None:
        bar_age_min = (
            pd.Timestamp.now(tz=last_ts.tzinfo) - last_ts
        ).total_seconds() / 60.0

    enriched = enrich_bars(bars, orb_minutes=cfg.orb_minutes)
    configure_htf_permission(cfg.htf_permission)
    configure_ote_params(cfg.ote)
    patterns = list(cfg.patterns)
    # AMD detector: tuned short-DTE recipe (Mon–Thu, reclaim, HTF, 10–12)
    if "amd" in patterns and cfg.amd.enabled:
        configure_amd_params(cfg.amd.detector_dict())
    elif "amd" in patterns:
        patterns = [p for p in patterns if p != "amd"]
    clear_ote_day_state()
    set_htf_cache(enriched)
    try:
        sig = best_signal(
            enriched,
            enabled=patterns,
            min_confidence=cfg.score_threshold,
            pattern_min_confidence=cfg.pattern_min_confidence,
        )
    finally:
        clear_htf_cache()

    journal = Journal(settings)
    broker = AlpacaBroker(settings, dry_run=dry_run)
    if settings.alpaca_api_key:
        broker.connect()
    account = broker.get_account()
    equity = float(account["equity"])
    positions = broker.list_positions()
    spy_opt_positions = [
        p
        for p in positions
        if len(str(p.get("symbol", ""))) >= 15
        and underlying_from_option_symbol(str(p["symbol"])) == cfg.symbol.upper()
    ]

    # Count today's SPY day entries from journal
    trades = journal.load_trades()
    today = date.today().isoformat()
    today_entries = 0
    if trades is not None and len(trades):
        t = trades.copy()
        if "timestamp" in t.columns and "ticker" in t.columns:
            ts = t["timestamp"].astype(str)
            tick = t["ticker"].astype(str).str.upper()
            side = t["side"].astype(str) if "side" in t.columns else pd.Series([""] * len(t))
            today_entries = int(
                (
                    ts.str.startswith(today)
                    & (tick == cfg.symbol.upper())
                    & side.str.contains("buy", case=False, na=False)
                ).sum()
            )

    summary: dict = {
        "mode": "spy_day",
        "paper": settings.paper,
        "dry_run": dry_run,
        "gate_ok": gate_ok,
        "gate_reason": gate_reason,
        "equity": equity,
        "n_bars": len(enriched),
        "last_bar": str(enriched.index[-1]),
        "bar_age_min": bar_age_min,
        "signal": None,
        "action": "flat",
    }
    if bar_age_min is not None and bar_age_min > STALE_BARS_MIN:
        summary["stale_bars"] = True
        print(
            f"[scan_spy_day] WARNING stale bars: last={enriched.index[-1]} "
            f"({bar_age_min:.0f} min old) — signals will be missed",
            file=sys.stderr,
        )

    if sig is None:
        journal.log_decision(
            ticker=cfg.symbol,
            action="flat",
            reason="no_pattern",
            details=f"last_bar={enriched.index[-1]}",
        )
        summary["action"] = "flat"
        return summary

    summary["signal"] = {
        "pattern": sig.pattern,
        "side": sig.side,
        "confidence": sig.confidence,
        "reason": sig.reason,
        "spot": sig.spot,
        "bar_time": str(sig.bar_time),
    }

    if len(spy_opt_positions) >= cfg.max_open_positions:
        journal.log_decision(
            ticker=cfg.symbol,
            action="skip",
            reason="already_in_position",
            confidence=sig.confidence,
            details=sig.reason,
        )
        summary["action"] = "skip_position"
        return summary

    books = classify_spy_option_books(
        spy_opt_positions,
        spy=cfg.symbol.upper(),
        amd_min_dte=cfg.amd.min_dte,
        amd_max_dte=cfg.amd.max_dte,
    )
    # Block ORB retest stacking on same-side 0DTE (was causing 6-lot doubles).
    if sig.pattern == "orb_retest" and sig.side in books["0dte"]:
        journal.log_decision(
            ticker=cfg.symbol,
            action="skip",
            reason="retest_same_side_open",
            confidence=sig.confidence,
            details=f"already long 0DTE {sig.side}",
        )
        summary["action"] = "skip_retest_stack"
        return summary
    # Block AMD vs opposing (or any) 0DTE book — no call+put fights.
    if sig.pattern == "amd" and books["0dte"]:
        journal.log_decision(
            ticker=cfg.symbol,
            action="skip",
            reason="amd_blocked_by_0dte",
            confidence=sig.confidence,
            details=f"0dte_open={sorted(books['0dte'])}",
        )
        summary["action"] = "skip_amd_vs_0dte"
        return summary
    # Block new 0DTE if AMD already open (one book policy + no opposing).
    if sig.pattern in {"orb", "orb_retest", "power_hour"} and books["amd"]:
        journal.log_decision(
            ticker=cfg.symbol,
            action="skip",
            reason="0dte_blocked_by_amd",
            confidence=sig.confidence,
            details=f"amd_open={sorted(books['amd'])}",
        )
        summary["action"] = "skip_0dte_vs_amd"
        return summary

    # AMD is short-DTE only; skip Fri already in detector. Extra guards.
    if sig.pattern == "amd":
        if not cfg.amd.enabled:
            journal.log_decision(
                ticker=cfg.symbol,
                action="skip",
                reason="amd_disabled",
                confidence=sig.confidence,
            )
            summary["action"] = "skip_amd_disabled"
            return summary
        if datetime.now(ET).weekday() == 4 and cfg.amd.skip_friday:
            journal.log_decision(
                ticker=cfg.symbol,
                action="skip",
                reason="amd_skip_friday",
                confidence=sig.confidence,
            )
            summary["action"] = "skip_amd_friday"
            return summary
        # One short-DTE AMD book at a time (ORB 0DTE may still be open)
        for p in spy_opt_positions:
            dte = _option_dte(str(p.get("symbol", "")))
            if dte is not None and cfg.amd.min_dte <= dte <= cfg.amd.max_dte:
                journal.log_decision(
                    ticker=cfg.symbol,
                    action="skip",
                    reason="amd_already_open",
                    confidence=sig.confidence,
                    details=str(p.get("symbol")),
                )
                summary["action"] = "skip_amd_open"
                return summary

    if today_entries >= cfg.max_trades_per_day:
        journal.log_decision(
            ticker=cfg.symbol,
            action="skip",
            reason="max_trades_per_day",
            confidence=sig.confidence,
        )
        summary["action"] = "skip_max_trades"
        return summary

    # Override risk notional for this lane
    settings.raw.setdefault("risk", {})["max_notional_per_trade"] = cfg.max_notional_per_trade
    settings.raw["risk"]["max_contracts_per_trade"] = cfg.max_contracts_per_trade
    if cfg.scale_out.enabled:
        # Don't let the 1% equity cap shrink a 3-lot AMD book.
        settings.raw["risk"]["max_risk_per_trade_pct"] = max(
            float(settings.raw["risk"].get("max_risk_per_trade_pct", 0.01)),
            0.02,
        )
    risk = RiskManager(settings)
    risk.state.open_positions = len(positions)

    ok_day, day_reasons = risk.check_loss_limits(equity)
    if not ok_day:
        summary["action"] = "skip_daily_loss"
        summary["reasons"] = day_reasons
        return summary

    option_type = sig.side  # call | put
    spot = float(sig.spot)
    is_amd = sig.pattern == "amd"
    min_dte = cfg.amd.min_dte if is_amd else cfg.min_dte
    max_dte = cfg.amd.max_dte if is_amd else cfg.max_dte
    prefer_dte = cfg.amd.target_dte if is_amd else None
    # Short-dated AMD mids are richer; allow up to $5 mid while keeping floor
    max_mid = 5.0 if is_amd else cfg.max_mid_price
    min_mid = cfg.min_mid_price
    allow_synthetic = dry_run and not settings.alpaca_api_key
    raw = broker.get_option_contracts(cfg.symbol, option_type, min_dte, max_dte)
    if raw:
        candidates = contracts_from_alpaca_payload(raw, cfg.symbol, option_type, spot)
        candidates = [c for c in candidates if min_dte <= c.dte <= max_dte]
        _enrich_quotes(broker, candidates)
    elif allow_synthetic:
        candidates = synthetic_candidates_for_backtest(
            cfg.symbol,
            spot,
            option_type,
            date.today(),
            target_dte=prefer_dte if prefer_dte is not None else max(max_dte, 0),
            premium_pct=0.008 if is_amd else 0.005,
        )
    else:
        journal.log_decision(
            ticker=cfg.symbol,
            action="skip",
            reason="no_contracts",
            confidence=sig.confidence,
            details=sig.reason,
        )
        summary["action"] = "skip_no_contracts"
        return summary

    contract, notes = select_contract(
        candidates,
        spot=spot,
        min_dte=min_dte,
        max_dte=max(max_dte, 0),
        otm_pct_min=cfg.otm_pct_min,
        otm_pct_max=cfg.otm_pct_max,
        min_open_interest=cfg.min_open_interest,
        min_option_volume=cfg.min_option_volume,
        max_spread_pct=0.25,
        max_mid_price=max_mid,
        min_mid_price=min_mid,
        target_delta_min=cfg.target_delta_min,
        target_delta_max=cfg.target_delta_max,
        target_delta=cfg.target_delta,
        target_dte=prefer_dte,
        rank="atm",
    )
    # 0DTE select may fail if max_dte=0 and synthetic dte=0 — relax when empty
    if contract is None and allow_synthetic:
        candidates = synthetic_candidates_for_backtest(
            cfg.symbol,
            spot,
            option_type,
            date.today(),
            target_dte=max(prefer_dte or 1, 1),
            premium_pct=0.008 if is_amd else 0.005,
        )
        contract, notes = select_contract(
            candidates,
            spot=spot,
            min_dte=0,
            max_dte=max(prefer_dte or 1, 1),
            otm_pct_min=cfg.otm_pct_min,
            otm_pct_max=cfg.otm_pct_max,
            min_open_interest=0,
            min_option_volume=0,
            max_spread_pct=0.5,
            max_mid_price=max_mid,
            min_mid_price=None,
            target_delta_min=cfg.target_delta_min,
            target_delta_max=cfg.target_delta_max,
            target_delta=cfg.target_delta,
            target_dte=prefer_dte,
            rank="atm",
        )

    if contract is None:
        journal.log_decision(
            ticker=cfg.symbol,
            action="skip",
            reason="contract_filter",
            confidence=sig.confidence,
            details="; ".join(notes) if notes else sig.reason,
        )
        summary["action"] = "skip_contract"
        summary["notes"] = notes
        return summary

    # Hard veto: still refuse sub-floor premiums even if filters slipped
    if contract.mid < cfg.min_mid_price:
        journal.log_decision(
            ticker=cfg.symbol,
            action="skip",
            reason="premium_too_cheap",
            confidence=sig.confidence,
            contract=contract.symbol,
            details=f"mid={contract.mid:.2f} < min_mid={cfg.min_mid_price:.2f}",
        )
        summary["action"] = "skip_cheap_contract"
        summary["contract"] = contract.symbol
        summary["mid"] = contract.mid
        return summary

    sizing = risk.size_order(equity, premium=contract.mid)
    if not sizing.allowed:
        summary["action"] = "skip_risk"
        summary["reasons"] = sizing.reasons
        return summary
    if cfg.scale_out.enabled:
        want = max(1, int(cfg.scale_out.qty_for(sig.pattern)))
        cost = float(contract.mid) * 100.0 * want
        if cost > cfg.max_notional_per_trade:
            journal.log_decision(
                ticker=cfg.symbol,
                action="skip",
                reason="scale_out_notional",
                confidence=sig.confidence,
                contract=contract.symbol,
                details=f"{want}-lot notional ${cost:.0f} > cap ${cfg.max_notional_per_trade:.0f}",
            )
            summary["action"] = "skip_scale_out_notional"
            summary["reasons"] = [f"{want}-lot notional ${cost:.0f} exceeds ${cfg.max_notional_per_trade:.0f}"]
            return summary
        from dataclasses import replace as _dc_replace

        sizing = _dc_replace(sizing, qty=want, reasons=["scale_out_qty"])

    limit_price = contract.mid if contract.mid > 0 else contract.ask
    order = OrderRequest(
        symbol=contract.symbol,
        qty=sizing.qty,
        side="buy",
        limit_price=limit_price,
        position_intent="buy_to_open",
    )
    result = broker.submit_option_order(order)
    journal.log_decision(
        ticker=cfg.symbol,
        action="order",
        signal=sig.side,
        pattern=sig.pattern,
        confidence=sig.confidence,
        contract=contract.symbol,
        qty=sizing.qty,
        limit_price=limit_price,
        dry_run=result.dry_run,
        status=result.status,
        order_id=result.order_id,
        details=f"{sig.pattern}: {sig.reason}",
        min_dte=min_dte,
        max_dte=max_dte,
        target_dte=prefer_dte,
    )
    journal.log_trade(
        ticker=cfg.symbol,
        side="buy_to_open",
        contract=contract.symbol,
        qty=sizing.qty,
        limit_price=limit_price,
        status=result.status,
        dry_run=result.dry_run,
        signal=sig.side,
        pattern=sig.pattern,
        confidence=sig.confidence,
        pnl=0.0,
        entry_premium=limit_price,
        expiration=str(contract.expiration),
        dte=contract.dte,
    )
    summary["action"] = "order"
    summary["pattern"] = sig.pattern
    summary["contract"] = contract.symbol
    summary["dte"] = contract.dte
    summary["qty"] = sizing.qty
    summary["limit_price"] = round(limit_price, 2)
    summary["order_status"] = result.status
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="SPY 0DTE day pattern scan")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="Refresh 5m bars before scan")
    args = parser.parse_args()
    dry = True
    if args.submit:
        dry = False
    if args.dry_run:
        dry = True
    print(
        json.dumps(
            run_spy_day_scan(dry_run=dry, submit=args.submit, refresh=args.refresh),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
