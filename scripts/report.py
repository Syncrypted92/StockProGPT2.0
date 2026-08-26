"""Report journal summary and optional backtest metrics."""

from __future__ import annotations

import argparse
import json

from stockpro.config import load_settings
from stockpro.data import download_universe
from stockpro.journal import Journal
from stockpro.backtest import run_walk_forward_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="StockPro reports")
    parser.add_argument("--backtest", action="store_true", help="Run walk-forward backtest")
    parser.add_argument("--tickers", nargs="*", help="Tickers for backtest")
    args = parser.parse_args()

    settings = load_settings()
    journal = Journal(settings)
    print("Journal summary:")
    print(json.dumps(journal.summary(), indent=2))

    if args.backtest:
        uni = settings.get("universe", default={}) or {}
        bt = settings.get("backtest", default={}) or {}
        feat = settings.get("features", default={}) or {}
        model_cfg = settings.get("model", default={}) or {}
        risk = settings.get("risk", default={}) or {}
        tickers = args.tickers or uni.get("tickers", ["SPY"])[:4]
        frames = download_universe(
            tickers,
            lookback_days=uni.get("lookback_days"),
            start=uni.get("history_start", "2000-01-01"),
        )
        result = run_walk_forward_backtest(
            frames,
            initial_equity=float(bt.get("initial_equity", 100000)),
            spread_penalty_pct=float(bt.get("spread_penalty_pct", 0.04)),
            option_premium_pct_of_spot=float(bt.get("option_premium_pct_of_spot", 0.02)),
            train_days=int(bt.get("walk_forward_train_days", 252)),
            test_days=int(bt.get("walk_forward_test_days", 42)),
            horizon=int(feat.get("forward_horizon_days", 5)),
            flat_threshold=float(feat.get("flat_threshold", 0.005)),
            probability_threshold=float(model_cfg.get("probability_threshold", 0.55)),
            max_risk_per_trade_pct=float(risk.get("max_risk_per_trade_pct", 0.01)),
            max_notional_per_trade=float(risk.get("max_notional_per_trade", 500)),
            max_contracts=int(risk.get("max_contracts_per_trade", 1)),
        )
        print("Backtest metrics:")
        print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
