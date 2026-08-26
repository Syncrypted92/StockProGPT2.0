"""Scan universe, generate signals, select contracts, apply risk, paper/live trade."""

from __future__ import annotations

import argparse
import json
from datetime import date

from stockpro.accuracy import log_prediction
from stockpro.broker import AlpacaBroker, OrderRequest
from stockpro.broker.live import assert_live_allowed
from stockpro.config import ROOT, load_settings
from stockpro.data import download_ohlcv
from stockpro.journal import Journal
from stockpro.models import latest_feature_row, load_model
from stockpro.news import (
    combined_rank_score,
    entry_news_veto,
    log_news_features,
    news_feature_dict_for_model,
    score_ticker_news,
)
from stockpro.options import (
    contracts_from_alpaca_payload,
    filter_underlying_liquidity,
    select_contract,
    synthetic_candidates_for_backtest,
)
from stockpro.risk import (
    RiskManager,
    open_underlyings,
    stop_cooldown_tickers,
    ticker_stop_rate,
)
from stockpro.signals import Signal, score_to_signal


def _enrich_with_quotes(broker: AlpacaBroker, candidates: list) -> None:
    symbols = [c.symbol for c in candidates if c.symbol and "_SYN_" not in c.symbol]
    if not symbols:
        return
    quotes = broker.get_option_quotes(symbols[:300])
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


def _shortlist_for_quotes(candidates: list, spot: float, opt_cfg: dict) -> list:
    """Reduce chain to OTM/DTE/OI band before requesting quotes."""
    min_dte = int(opt_cfg.get("min_dte", 14))
    max_dte = int(opt_cfg.get("max_dte", 45))
    otm_min = float(opt_cfg.get("otm_pct_min", 0.01))
    otm_max = float(opt_cfg.get("otm_pct_max", 0.08))
    min_oi = int(opt_cfg.get("min_open_interest", 100))
    out = []
    for c in candidates:
        if not (min_dte <= c.dte <= max_dte):
            continue
        if c.open_interest >= 0:
            if c.open_interest < min_oi and c.volume < int(opt_cfg.get("min_option_volume", 10)):
                continue
        if c.option_type == "call":
            otm = (c.strike - spot) / spot
        else:
            otm = (spot - c.strike) / spot
        if otm_min <= otm <= otm_max:
            out.append(c)
    return out


def run_scan(
    dry_run_override: bool | None = None,
    submit: bool = False,
    zerodte: bool = False,
) -> dict:
    settings = load_settings()
    uni = dict(settings.get("universe", default={}) or {})
    opt_cfg = dict(settings.get("options", default={}) or {})
    model_cfg = dict(settings.get("model", default={}) or {})
    risk_cfg = dict(settings.get("risk", default={}) or {})
    zd = dict(settings.get("zerodte", default={}) or {})

    mode = "zerodte" if zerodte else "swing"
    if zerodte:
        if not zd.get("enabled", False):
            raise RuntimeError("zerodte.enabled is false in config/settings.yaml")
        uni["tickers"] = list(zd.get("tickers") or ["SPY", "QQQ", "IWM"])
        opt_cfg["min_dte"] = int(zd.get("min_dte", 0))
        opt_cfg["max_dte"] = int(zd.get("max_dte", 1))
        opt_cfg["max_mid_price"] = float(zd.get("max_mid_price", 3.0))
        opt_cfg["otm_pct_min"] = float(zd.get("otm_pct_min", 0.001))
        opt_cfg["otm_pct_max"] = float(zd.get("otm_pct_max", 0.015))
        # 0DTE chains are liquid; loosen OI a bit for same-day contracts
        opt_cfg["min_open_interest"] = int(zd.get("min_open_interest", 50))
        opt_cfg["min_option_volume"] = int(zd.get("min_option_volume", 10))
        model_cfg["probability_threshold"] = float(zd.get("probability_threshold", 0.65))
        model_cfg["max_new_entries_per_scan"] = int(zd.get("max_new_entries_per_scan", 1))
        risk_cfg["max_notional_per_trade"] = float(zd.get("max_notional_per_trade", 300.0))
        settings.raw.setdefault("risk", {}).update(risk_cfg)

    dry_run = risk_cfg.get("dry_run", True) if dry_run_override is None else dry_run_override
    if submit:
        dry_run = False

    if not settings.paper:
        assert_live_allowed(settings)
        from stockpro.broker.live import live_risk_overrides

        for k, v in live_risk_overrides().items():
            risk_cfg[k] = v
        settings.raw.setdefault("risk", {}).update(risk_cfg)

    if settings.trading_halted:
        raise RuntimeError("TRADING_HALTED=true — refusing to scan/trade")

    artifact_dir = ROOT / model_cfg.get("artifact_dir", "artifacts")
    model, meta = load_model(artifact_dir)
    threshold = float(model_cfg.get("probability_threshold", 0.65))
    max_entries = int(model_cfg.get("max_new_entries_per_scan", 2))
    history_start = uni.get("history_start", "2000-01-01")
    lookback = uni.get("lookback_days")
    one_per_underlying = bool(risk_cfg.get("one_position_per_underlying", True))
    cooldown_days = int(risk_cfg.get("cooldown_days_after_stop", 3))
    stop_flag_thr = float(risk_cfg.get("stop_rate_flag_threshold", 0.70))
    stop_lookback = int(risk_cfg.get("stop_rate_lookback_trades", 5))
    news_cfg = dict(settings.get("news", default={}) or {})
    news_scoring = bool(news_cfg.get("scoring_enabled", False))
    news_weight = float(news_cfg.get("news_weight", 0.25))
    min_combined = float(news_cfg.get("min_combined_score", 0.0))
    skip_news_conflict = bool(news_cfg.get("skip_news_conflict", True))
    model_uses_news = bool(meta.get("use_news_features"))
    model_feature_cols = list(meta.get("feature_columns") or [])

    broker = AlpacaBroker(settings, dry_run=dry_run)
    if settings.alpaca_api_key:
        broker.connect()
    account = broker.get_account()
    equity = float(account["equity"])
    positions = broker.list_positions()
    held_underlyings = open_underlyings(positions)

    risk = RiskManager(settings)
    risk.state.open_positions = len(positions)
    journal = Journal(settings)
    trades = journal.load_trades()
    cooldown = stop_cooldown_tickers(trades, cooldown_days=cooldown_days, as_of=date.today())

    tickers = uni.get("tickers", ["SPY"])
    allow_synthetic = dry_run and not settings.alpaca_api_key

    spy_close = None
    try:
        spy_df = download_ohlcv("SPY", lookback_days=lookback, start=history_start)
        if spy_df is not None and not spy_df.empty:
            spy_close = spy_df["Close"]
    except Exception:  # noqa: BLE001
        spy_close = None

    # Pass 1: score every ticker and collect actionable candidates ranked by confidence
    scored: list[dict] = []
    skipped: list[dict] = []

    for ticker in tickers:
        ohlcv = download_ohlcv(
            ticker,
            lookback_days=lookback,
            start=history_start,
        )
        as_of = str(ohlcv.index.max().date())
        # 0DTE ETFs are always liquid enough; skip harsh dollar-volume gate
        if zerodte:
            from stockpro.options import LiquidityReport

            last_price = float(ohlcv["Close"].iloc[-1])
            liq = LiquidityReport(
                ticker=ticker,
                passed=True,
                last_price=last_price,
                avg_volume=float(ohlcv["Volume"].tail(20).mean()),
                avg_dollar_volume=float((ohlcv["Close"] * ohlcv["Volume"]).tail(20).mean()),
                reasons=[],
            )
        else:
            liq = filter_underlying_liquidity(
                ticker,
                ohlcv,
                min_price=float(uni.get("min_price", 10)),
                min_avg_volume=float(uni.get("min_avg_volume", 2_000_000)),
                min_avg_dollar_volume=float(uni.get("min_avg_dollar_volume", 50_000_000)),
                volume_lookback_days=int(uni.get("volume_lookback_days", 20)),
            )
        if not liq.passed:
            row = {
                "ticker": ticker,
                "action": "skip_liquidity",
                "reasons": liq.reasons,
                "as_of": as_of,
                "bars": len(ohlcv),
            }
            skipped.append(row)
            journal.log_decision(ticker=ticker, action="skip", reason="liquidity", details="; ".join(liq.reasons))
            continue

        features = latest_feature_row(ohlcv, spy_close=spy_close)
        # Score news early so model can consume Tiingo features when trained with them
        news_score_obj = None
        if news_scoring or model_uses_news:
            news_score_obj = score_ticker_news(
                settings,
                ticker,
                as_of=date.today(),
                signal_direction=0,
            )
            log_news_features(settings, news_score_obj, as_of=date.today())
            if model_uses_news:
                features = latest_feature_row(
                    ohlcv,
                    news_features=news_feature_dict_for_model(news_score_obj),
                    feature_columns=model_feature_cols or None,
                    spy_close=spy_close,
                )
            elif model_feature_cols:
                features = latest_feature_row(
                    ohlcv,
                    feature_columns=model_feature_cols,
                    spy_close=spy_close,
                )

        signal = score_to_signal(model, features, ticker, probability_threshold=threshold)
        if not zerodte:
            log_prediction(
                settings,
                as_of=as_of,
                ticker=ticker,
                signal=signal.signal,
                predicted_class=signal.predicted_class,
                confidence=signal.confidence,
                spot=liq.last_price,
            )
        base = {
            "ticker": ticker,
            "as_of": as_of,
            "bars": len(ohlcv),
            "spot": liq.last_price,
            "signal": signal.signal.value,
            "confidence": round(signal.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in signal.probabilities.items()},
            "predicted_class": signal.predicted_class,
            "mode": mode,
        }

        combined = signal.confidence
        if news_score_obj is not None and news_scoring:
            combined = combined_rank_score(
                signal.confidence,
                news_score_obj,
                signal.predicted_class,
                news_weight,
            )

        if signal.signal == Signal.FLAT:
            row = {**base, "action": "flat"}
            skipped.append(row)
            journal.log_decision(
                ticker=ticker,
                action="flat",
                confidence=signal.confidence,
                predicted_class=signal.predicted_class,
                probabilities=json.dumps(signal.probabilities),
            )
            if news_score_obj is not None and news_score_obj.score >= 0.4:
                journal.log_decision(
                    ticker=ticker,
                    action="watchlist",
                    reason="strong_news_flat_technical",
                    confidence=signal.confidence,
                    details=f"news_score={news_score_obj.score:.2f}; {news_score_obj.top_headline[:120]}",
                )
            continue

        scored.append(
            {
                **base,
                "ohlcv": ohlcv,
                "signal_obj": signal,
                "liq": liq,
                "news_score": round(news_score_obj.score, 4) if news_score_obj else 0.0,
                "combined_score": round(combined, 4),
                "top_headline": news_score_obj.top_headline if news_score_obj else "",
            }
        )

    # Highest combined score first (falls back to confidence when news scoring off)
    sort_key = "combined_score" if news_scoring else "confidence"
    scored.sort(key=lambda r: r[sort_key], reverse=True)
    confidence_board = [
        {
            "rank": i + 1,
            "ticker": r["ticker"],
            "signal": r["signal"],
            "confidence": r["confidence"],
            "news_score": r.get("news_score"),
            "combined_score": r.get("combined_score", r["confidence"]),
            "probabilities": r["probabilities"],
            "as_of": r["as_of"],
            "selected_for_trade": i < max_entries,
        }
        for i, r in enumerate(scored)
    ]

    results: list[dict] = []
    results.extend(skipped)

    # Pass 2: only top-N confident directional signals become orders
    for i, item in enumerate(scored):
        ticker = item["ticker"]
        signal = item["signal_obj"]
        liq = item["liq"]
        if i >= max_entries:
            results.append(
                {
                    "ticker": ticker,
                    "action": "skipped_lower_confidence",
                    "signal": item["signal"],
                    "confidence": item["confidence"],
                    "probabilities": item["probabilities"],
                    "as_of": item["as_of"],
                }
            )
            journal.log_decision(
                ticker=ticker,
                action="skip",
                reason="lower_confidence",
                confidence=item["confidence"],
            )
            continue

        if one_per_underlying and ticker.upper() in held_underlyings:
            results.append(
                {
                    "ticker": ticker,
                    "action": "skip_open_underlying",
                    "confidence": item["confidence"],
                    "signal": item["signal"],
                }
            )
            journal.log_decision(
                ticker=ticker,
                action="skip",
                reason="open_underlying",
                confidence=item["confidence"],
                details="already have open option on this underlying",
            )
            continue

        if ticker.upper() in cooldown:
            stop_day = cooldown[ticker.upper()]
            results.append(
                {
                    "ticker": ticker,
                    "action": "skip_cooldown",
                    "confidence": item["confidence"],
                    "signal": item["signal"],
                    "stop_day": stop_day.isoformat(),
                    "cooldown_days": cooldown_days,
                }
            )
            journal.log_decision(
                ticker=ticker,
                action="skip",
                reason="cooldown",
                confidence=item["confidence"],
                details=f"stop_loss on {stop_day.isoformat()}; {cooldown_days} trading-day cooldown",
            )
            continue

        rate = ticker_stop_rate(trades, ticker, lookback=stop_lookback)
        if rate is not None and rate >= stop_flag_thr:
            results.append(
                {
                    "ticker": ticker,
                    "action": "skip_stop_rate",
                    "confidence": item["confidence"],
                    "signal": item["signal"],
                    "stop_rate": round(rate, 2),
                    "lookback": stop_lookback,
                }
            )
            journal.log_decision(
                ticker=ticker,
                action="skip",
                reason="stop_rate",
                confidence=item["confidence"],
                details=f"stop_rate={rate:.0%} over last {stop_lookback} exits (>= {stop_flag_thr:.0%})",
            )
            continue

        veto = entry_news_veto(settings, ticker, as_of=date.today())
        if veto.blocked:
            results.append(
                {
                    "ticker": ticker,
                    "action": f"skip_{veto.reason}",
                    "confidence": item["confidence"],
                    "signal": item["signal"],
                    "detail": veto.detail,
                }
            )
            journal.log_decision(
                ticker=ticker,
                action="skip",
                reason=veto.reason,
                confidence=item["confidence"],
                details=veto.detail,
            )
            continue

        combined = float(item.get("combined_score", item["confidence"]))
        news_sc = float(item.get("news_score", 0.0))
        if combined < min_combined:
            results.append(
                {
                    "ticker": ticker,
                    "action": "skip_low_combined_score",
                    "confidence": item["confidence"],
                    "combined_score": combined,
                    "news_score": news_sc,
                }
            )
            journal.log_decision(
                ticker=ticker,
                action="skip",
                reason="low_combined_score",
                confidence=item["confidence"],
                details=f"combined={combined:.3f} news={news_sc:.3f}",
            )
            continue

        if skip_news_conflict and news_scoring:
            sig_dir = signal.predicted_class
            if (sig_dir == 1 and news_sc < -0.15) or (sig_dir == -1 and news_sc > 0.15):
                results.append(
                    {
                        "ticker": ticker,
                        "action": "skip_news_conflict",
                        "confidence": item["confidence"],
                        "news_score": news_sc,
                        "combined_score": combined,
                    }
                )
                journal.log_decision(
                    ticker=ticker,
                    action="skip",
                    reason="news_conflict",
                    confidence=item["confidence"],
                    details=f"signal={item['signal']} news_score={news_sc:.3f}",
                )
                continue

        option_type = "call" if signal.signal == Signal.BULLISH else "put"
        spot = liq.last_price
        raw_contracts = broker.get_option_contracts(
            ticker,
            option_type,
            int(opt_cfg.get("min_dte", 14)),
            int(opt_cfg.get("max_dte", 45)),
        )

        if raw_contracts:
            candidates = contracts_from_alpaca_payload(raw_contracts, ticker, option_type, spot)
            candidates = _shortlist_for_quotes(candidates, spot, opt_cfg)
            _enrich_with_quotes(broker, candidates)
        elif allow_synthetic:
            candidates = synthetic_candidates_for_backtest(ticker, spot, option_type, date.today())
        else:
            results.append(
                {
                    "ticker": ticker,
                    "action": "skip_no_contracts",
                    "confidence": item["confidence"],
                    "signal": item["signal"],
                }
            )
            continue

        contract, notes = select_contract(
            candidates,
            spot=spot,
            min_dte=int(opt_cfg.get("min_dte", 14)),
            max_dte=int(opt_cfg.get("max_dte", 45)),
            target_delta_min=float(opt_cfg.get("target_delta_min", 0.3)),
            target_delta_max=float(opt_cfg.get("target_delta_max", 0.5)),
            otm_pct_min=float(opt_cfg.get("otm_pct_min", 0.01)),
            otm_pct_max=float(opt_cfg.get("otm_pct_max", 0.08)),
            min_open_interest=int(opt_cfg.get("min_open_interest", 100)),
            min_option_volume=int(opt_cfg.get("min_option_volume", 10)),
            max_spread_pct=float(opt_cfg.get("max_spread_pct", 0.15)),
            max_mid_price=float(opt_cfg["max_mid_price"]) if opt_cfg.get("max_mid_price") is not None else None,
        )
        if contract is None:
            results.append(
                {
                    "ticker": ticker,
                    "action": "skip_contract",
                    "notes": notes,
                    "confidence": item["confidence"],
                    "signal": item["signal"],
                }
            )
            continue

        if not dry_run and "_SYN_" in contract.symbol:
            results.append({"ticker": ticker, "action": "skip_synthetic_blocked"})
            continue

        sizing = risk.size_order(equity, premium=contract.mid)
        if not sizing.allowed:
            results.append(
                {
                    "ticker": ticker,
                    "action": "skip_risk",
                    "reasons": sizing.reasons,
                    "confidence": item["confidence"],
                    "signal": item["signal"],
                    "contract": contract.symbol,
                }
            )
            continue

        limit_price = contract.mid if opt_cfg.get("prefer_mid_limit", True) else contract.ask
        if limit_price <= 0:
            results.append({"ticker": ticker, "action": "skip_bad_price"})
            continue

        order = OrderRequest(
            symbol=contract.symbol,
            qty=sizing.qty,
            side="buy",
            limit_price=limit_price,
            position_intent="buy_to_open",
        )
        order_result = broker.submit_option_order(order)
        if order_result.dry_run or order_result.submitted:
            risk.state.open_positions += 1
            held_underlyings.add(ticker.upper())

        journal.log_decision(
            ticker=ticker,
            action="order",
            signal=signal.signal.value,
            confidence=signal.confidence,
            probabilities=json.dumps(signal.probabilities),
            contract=contract.symbol,
            qty=sizing.qty,
            limit_price=limit_price,
            dry_run=order_result.dry_run,
            status=order_result.status,
            order_id=order_result.order_id,
            rank=i + 1,
            details=(
                f"combined={combined:.3f} news={news_sc:.3f}; "
                f"{item.get('top_headline', '')[:80]}"
            ),
        )
        journal.log_trade(
            ticker=ticker,
            side="buy_to_open",
            contract=contract.symbol,
            qty=sizing.qty,
            limit_price=limit_price,
            status=order_result.status,
            dry_run=order_result.dry_run,
            signal=signal.signal.value,
            confidence=signal.confidence,
            pnl=0.0,
            entry_premium=limit_price,
            expiration=str(contract.expiration),
        )
        results.append(
            {
                "ticker": ticker,
                "action": "order",
                "rank": i + 1,
                "signal": signal.signal.value,
                "confidence": item["confidence"],
                "probabilities": item["probabilities"],
                "contract": contract.symbol,
                "qty": sizing.qty,
                "limit_price": round(limit_price, 2),
                "bid": contract.bid,
                "ask": contract.ask,
                "dte": contract.dte,
                "status": order_result.status,
                "dry_run": order_result.dry_run,
                "as_of": item["as_of"],
            }
        )

    return {
        "paper": settings.paper,
        "dry_run": dry_run,
        "mode": mode,
        "equity": equity,
        "open_positions": len(positions),
        "held_underlyings": sorted(held_underlyings),
        "cooldown_tickers": {k: v.isoformat() for k, v in cooldown.items()},
        "options_trading_level": account.get("options_trading_level"),
        "probability_threshold": threshold,
        "confidence_board": confidence_board,
        "results": results,
        "model_meta": {
            k: meta.get(k)
            for k in (
                "test_accuracy",
                "hit_rate_directional",
                "trained_at",
                "model_name",
                "history_start",
                "tickers",
            )
            if k in meta
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan and trade directional options")
    parser.add_argument("--submit", action="store_true", help="Actually submit orders (disables dry_run)")
    parser.add_argument("--dry-run", action="store_true", help="Force dry-run")
    parser.add_argument(
        "--zerodte",
        action="store_true",
        help="0DTE/1DTE paper lane (SPY/QQQ/IWM, same-day expiries)",
    )
    args = parser.parse_args()
    dry = True if args.dry_run else (False if args.submit else None)
    summary = run_scan(dry_run_override=dry, submit=args.submit, zerodte=args.zerodte)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
