"""Train and persist the directional baseline model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from stockpro.config import ROOT, load_settings
from stockpro.data import download_universe
from stockpro.models import (
    optimize_directional_model,
    save_model,
    train_ab_comparison,
    train_directional_model,
)
from stockpro.news import (
    backfill_tiingo_news_features,
    load_news_features_table,
    tiingo_configured,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train StockPro directional model")
    parser.add_argument("--tickers", nargs="*", help="Override universe tickers")
    parser.add_argument(
        "--with-news",
        action="store_true",
        default=None,
        help="Force include Tiingo news features (default: on when TIINGO_API_KEY set)",
    )
    parser.add_argument(
        "--no-news",
        action="store_true",
        help="Train technical-only even if Tiingo key is present",
    )
    parser.add_argument(
        "--skip-backfill",
        action="store_true",
        help="Use existing news_features.csv without refreshing from Tiingo",
    )
    parser.add_argument(
        "--ab-test",
        action="store_true",
        help="Compare technical-only vs technical+news; do not save model",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Val-based search over lookback/flat/HGB/news; save the winner",
    )
    args = parser.parse_args()

    settings = load_settings()
    uni = settings.get("universe", default={}) or {}
    feat = settings.get("features", default={}) or {}
    model_cfg = settings.get("model", default={}) or {}
    news_cfg = settings.get("news", default={}) or {}

    tickers = args.tickers or uni.get("tickers", ["SPY"])
    history_start = uni.get("history_start", "2000-01-01")
    lookback = uni.get("lookback_days")

    use_news = tiingo_configured() and not args.no_news
    if args.with_news is True:
        use_news = True
    if args.no_news:
        use_news = False

    print(f"Downloading {len(tickers)} tickers from {history_start} to latest...")
    frames = download_universe(
        tickers,
        lookback_days=lookback,
        start=history_start,
    )
    if not frames:
        raise SystemExit("No market data downloaded")

    train_kw = dict(
        horizon=int(feat.get("forward_horizon_days", 5)),
        flat_threshold=float(feat.get("flat_threshold", 0.008)),
        train_ratio=float(model_cfg.get("train_ratio", 0.7)),
        val_ratio=float(model_cfg.get("val_ratio", 0.15)),
        binary=bool(feat.get("binary_labels", False)),
    )

    news_df = None
    need_news = use_news or args.ab_test or args.optimize
    if need_news:
        lookback_days = int(news_cfg.get("train_lookback_days", 90))
        if not args.skip_backfill and tiingo_configured():
            print(f"Backfilling Tiingo news features ({lookback_days}d)...")
            news_df = backfill_tiingo_news_features(
                settings, list(frames.keys()), lookback_days=lookback_days, persist=True
            )
        else:
            news_df = load_news_features_table(settings)
        if news_df is None or news_df.empty:
            print("Warning: no news features available; continuing without news.")
            use_news = False

    if args.ab_test:
        ab = train_ab_comparison(
            frames,
            settings=settings,
            price_lookback_days=int(news_cfg.get("train_price_lookback_days", 504)),
            **train_kw,
        )
        print(json.dumps(ab, indent=2, default=str))
        return

    if args.optimize or bool(model_cfg.get("optimize_on_train", False)):
        print("Optimizing model on validation directional hit...")
        model, result, _clean = optimize_directional_model(
            frames,
            news_df=news_df,
            horizon=train_kw["horizon"],
            train_ratio=train_kw["train_ratio"],
            val_ratio=train_kw["val_ratio"],
            verbose=True,
        )
        use_news = bool((result.best_params or {}).get("use_news_features", use_news))
    else:
        model, result, _clean = train_directional_model(
            frames,
            news_df=news_df,
            use_news_features=use_news,
            news_window_only=bool(news_cfg.get("train_on_news_window_only", False)),
            price_lookback_days=(
                int(news_cfg.get("train_price_lookback_days", 504)) if use_news else None
            ),
            news_boost=float(model_cfg.get("news_sample_boost", 3.0)),
            directional_boost=float(model_cfg.get("directional_sample_boost", 1.5)),
            **train_kw,
        )

    artifact_dir = ROOT / model_cfg.get("artifact_dir", "artifacts")
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "tickers": list(frames.keys()),
        "history_start": history_start,
        "history_end_by_ticker": {
            t: str(df.index.max().date()) for t, df in frames.items()
        },
        "bars_by_ticker": {t: len(df) for t, df in frames.items()},
        "model_name": result.model_name,
        "train_accuracy": result.train_accuracy,
        "val_accuracy": result.val_accuracy,
        "val_hit_rate_directional": result.val_hit_rate_directional,
        "test_accuracy": result.test_accuracy,
        "hit_rate_directional": result.hit_rate_directional,
        "n_train": result.n_train,
        "n_val": result.n_val,
        "n_test": result.n_test,
        "feature_columns": result.feature_columns,
        "use_news_features": use_news,
        "news_rows": int(len(news_df)) if news_df is not None else 0,
        "tiingo_enabled": tiingo_configured(),
        "optimized": bool(args.optimize or model_cfg.get("optimize_on_train", False)),
        "best_params": result.best_params,
        "top_trials": (result.optimization_trials or [])[:8],
        "classification_report": result.report,
        "feature_importance": result.feature_importance,
    }
    path = save_model(model, meta, artifact_dir)
    print(f"Saved model to {path}")
    skip_keys = {"classification_report", "feature_importance", "top_trials"}
    print(json.dumps({k: meta[k] for k in meta if k not in skip_keys}, indent=2, default=str))
    if result.optimization_trials:
        print("\nTop validation trials:")
        for t in result.optimization_trials[:5]:
            print(
                f"  news={t['use_news']} lb={t['lookback']} flat={t['flat_threshold']} "
                f"val_hit={t['val_hit']:.3f} test_hit={t['test_hit']:.3f}"
            )
    if result.feature_importance:
        top = sorted(result.feature_importance.items(), key=lambda x: -x[1])[:12]
        print("\nTop feature importance (permutation):")
        for name, imp in top:
            print(f"  {name}: {imp:.4f}")
    print(result.report)


if __name__ == "__main__":
    main()
