"""End-to-end system optimize: model (features/horizon) + trade-layer expectancy."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stockpro.backtest import optimize_trade_layer
from stockpro.config import ROOT, load_settings
from stockpro.data import download_universe
from stockpro.models import optimize_directional_model, save_model
from stockpro.news import load_news_features_table, tiingo_configured


def _trim_frames(frames: dict[str, pd.DataFrame], years: float = 6.0) -> dict[str, pd.DataFrame]:
    out = {}
    for t, df in frames.items():
        if df.empty:
            continue
        cutoff = df.index.max() - pd.Timedelta(days=int(365.25 * years))
        out[t] = df.loc[df.index >= cutoff].copy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize model + trade layer for expectancy")
    parser.add_argument("--skip-download-cache", action="store_true", help="unused placeholder")
    parser.add_argument("--years", type=float, default=6.0, help="History years for trade-layer WF")
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--trade-only", action="store_true")
    parser.add_argument("--no-write-settings", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    uni = settings.get("universe", default={}) or {}
    feat = settings.get("features", default={}) or {}
    model_cfg = settings.get("model", default={}) or {}
    risk = settings.get("risk", default={}) or {}
    opt = settings.get("options", default={}) or {}
    bt = settings.get("backtest", default={}) or {}

    tickers = uni.get("tickers", ["SPY"])
    history_start = uni.get("history_start", "2000-01-01")
    print(f"Downloading {len(tickers)} tickers...")
    frames = download_universe(tickers, start=history_start)
    if not frames:
        raise SystemExit("No market data")

    news_df = load_news_features_table(settings) if tiingo_configured() else None
    artifact_dir = ROOT / model_cfg.get("artifact_dir", "artifacts")
    settings_path = ROOT / "config" / "settings.yaml"

    model = None
    result = None
    best_params: dict = {}

    if not args.trade_only:
        print("\n=== Phase 1: model optimize (regime/RS features, horizons, binary) ===")
        model, result, _ = optimize_directional_model(
            frames,
            news_df=news_df,
            train_ratio=float(model_cfg.get("train_ratio", 0.7)),
            val_ratio=float(model_cfg.get("val_ratio", 0.15)),
            verbose=True,
            system=True,
        )
        best_params = dict(result.best_params or {})
        meta = {
            "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "tickers": list(frames.keys()),
            "history_start": history_start,
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
            "use_news_features": bool(best_params.get("use_news_features", False)),
            "optimized": True,
            "system_optimize": True,
            "best_params": best_params,
            "top_trials": (result.optimization_trials or [])[:10],
            "classification_report": result.report,
            "feature_importance": result.feature_importance,
            "tiingo_enabled": tiingo_configured(),
        }
        path = save_model(model, meta, artifact_dir)
        print(f"Saved model to {path}")
    else:
        from stockpro.models import load_model

        model, meta = load_model(artifact_dir)
        best_params = dict(meta.get("best_params") or {})
        result = None

    trade_summary = None
    if not args.model_only:
        print("\n=== Phase 2: trade-layer expectancy grid (walk-forward + TP/SL path) ===")
        wf_frames = _trim_frames(frames, years=args.years)
        # Speed: mega-caps + liquid names only for trade grid
        prefer = [
            t
            for t in [
                "SPY",
                "QQQ",
                "IWM",
                "AAPL",
                "MSFT",
                "NVDA",
                "AMD",
                "XLF",
                "XLE",
                "BAC",
                "F",
                "INTC",
            ]
            if t in wf_frames
        ]
        if len(prefer) >= 6:
            wf_frames = {t: wf_frames[t] for t in prefer}

        hgb_keys = (
            "max_depth",
            "learning_rate",
            "min_samples_leaf",
            "l2_regularization",
            "max_iter",
        )
        model_params = {k: best_params[k] for k in hgb_keys if k in best_params}
        horizon = int(best_params.get("horizon", feat.get("forward_horizon_days", 5)))
        flat = float(best_params.get("flat_threshold", feat.get("flat_threshold", 0.008)))
        binary = bool(best_params.get("binary", False))
        spy = wf_frames["SPY"]["Close"] if "SPY" in wf_frames else None

        trade_summary = optimize_trade_layer(
            wf_frames,
            horizon=horizon,
            flat_threshold=flat,
            train_days=int(bt.get("walk_forward_train_days", 180)),
            test_days=int(bt.get("walk_forward_test_days", 30)),
            spy_close=spy,
            binary_labels=binary,
            model_params=model_params or None,
            min_trades=30,
            verbose=True,
        )
        best_trade = trade_summary["best"]
        print("\nTrade-layer winner:")
        print(json.dumps(best_trade, indent=2, default=str))

        # Compare to prior defaults using the same cached-grid trials when present
        baseline_trial = next(
            (
                t
                for t in trade_summary.get("trials", [])
                if abs(t.get("probability_threshold", 0) - float(model_cfg.get("probability_threshold", 0.65))) < 1e-9
                and abs(t.get("profit_target_pct", 0) - float(risk.get("profit_target_pct", 0.40))) < 1e-9
                and abs(t.get("stop_loss_pct", 0) - float(risk.get("stop_loss_pct", 0.25))) < 1e-9
                and t.get("min_dte") == int(opt.get("min_dte", 7))
            ),
            None,
        )
        if baseline_trial:
            print(
                f"\nBaseline-like E={baseline_trial['expectancy']:.2f} "
                f"PF={baseline_trial['profit_factor']:.2f} n={baseline_trial['n_trades']}"
            )
        print(
            f"Optimized E={best_trade['expectancy']:.2f} "
            f"PF={best_trade['profit_factor']:.2f} n={best_trade['n_trades']}"
        )

        (artifact_dir / "trade_layer_optimize.json").write_text(
            json.dumps(trade_summary, indent=2, default=str),
            encoding="utf-8",
        )

        if not args.no_write_settings:
            # Section-aware patches (avoid zerodte.* collisions)
            text = settings_path.read_text(encoding="utf-8")
            replacements = [
                (r"(?m)^(features:\n(?:.*\n)*?\s*forward_horizon_days:\s*).*$", rf"\g<1>{horizon}"),
                (r"(?m)^(model:\n(?:.*\n)*?\s*probability_threshold:\s*).*$", rf"\g<1>{best_trade['probability_threshold']:.2f}"),
                (r"(?m)^(risk:\n(?:.*\n)*?\s*profit_target_pct:\s*).*$", rf"\g<1>{best_trade['profit_target_pct']:.2f}"),
                (r"(?m)^(risk:\n(?:.*\n)*?\s*stop_loss_pct:\s*).*$", rf"\g<1>{best_trade['stop_loss_pct']:.2f}"),
                (r"(?m)^(options:\n(?:.*\n)*?\s*min_dte:\s*).*$", rf"\g<1>{best_trade['min_dte']}"),
                (r"(?m)^(options:\n(?:.*\n)*?\s*max_dte:\s*).*$", rf"\g<1>{best_trade['max_dte']}"),
            ]
            for pattern, repl in replacements:
                text, n = re.subn(pattern, repl, text, count=1)
                if n == 0:
                    print(f"Warning: settings patch missed: {pattern[:40]}")
            settings_path.write_text(text, encoding="utf-8")
            print(f"Updated {settings_path}")

        meta_path = artifact_dir / "model_meta.json"
        if meta_path.exists():
            meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
            meta_obj["trade_layer"] = best_trade
            if baseline_trial:
                meta_obj["trade_layer_baseline"] = baseline_trial
            meta_path.write_text(json.dumps(meta_obj, indent=2, default=str), encoding="utf-8")

    out = {
        "best_params": best_params,
        "model": None
        if result is None
        else {
            "test_accuracy": result.test_accuracy,
            "hit_rate_directional": result.hit_rate_directional,
            "val_hit_rate_directional": result.val_hit_rate_directional,
        },
        "trade_layer": None if trade_summary is None else trade_summary["best"],
    }
    print("\n=== Summary ===")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
