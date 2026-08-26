"""Directional baseline model with chronological splits and val-based optimization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stockpro.features import (
    ALL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    NEWS_FEATURE_COLUMNS,
    add_forward_labels,
    add_indicators,
    feature_matrix,
)
from stockpro.news import load_news_features_table


@dataclass
class TrainResult:
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    hit_rate_directional: float
    n_train: int
    n_val: int
    n_test: int
    feature_columns: list[str]
    report: str
    model_name: str = "HistGradientBoostingClassifier"
    feature_importance: dict[str, float] | None = None
    val_hit_rate_directional: float = float("nan")
    best_params: dict[str, Any] | None = None
    optimization_trials: list[dict[str, Any]] = field(default_factory=list)


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple:
    n = len(X)
    i_train = int(n * train_ratio)
    i_val = int(n * (train_ratio + val_ratio))
    return (
        X.iloc[:i_train],
        y.iloc[:i_train],
        X.iloc[i_train:i_val],
        y.iloc[i_train:i_val],
        X.iloc[i_val:],
        y.iloc[i_val:],
    )


def directional_hit_rate(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_pred != 0
    if not mask.any():
        return 0.0
    return float((y_pred[mask] == y_true[mask]).mean())


def build_model(
    *,
    use_news_features: bool = False,
    max_depth: int | None = None,
    learning_rate: float = 0.05,
    max_iter: int | None = None,
    l2_regularization: float | None = None,
    min_samples_leaf: int | None = None,
) -> Pipeline:
    """HistGradientBoosting with sensible defaults; overrides for tuning."""
    if max_depth is None:
        max_depth = 4 if use_news_features else 5
    if max_iter is None:
        max_iter = 200 if use_news_features else 300
    if l2_regularization is None:
        l2_regularization = 0.5 if use_news_features else 0.1
    if min_samples_leaf is None:
        min_samples_leaf = 40 if use_news_features else 20
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    random_state=42,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    l2_regularization=l2_regularization,
                    min_samples_leaf=min_samples_leaf,
                ),
            ),
        ]
    )


def prepare_training_frame(
    ohlcv: pd.DataFrame,
    horizon: int = 5,
    flat_threshold: float = 0.005,
    *,
    spy_close: pd.Series | None = None,
    binary: bool = False,
) -> pd.DataFrame:
    df = add_indicators(ohlcv, spy_close=spy_close)
    df = add_forward_labels(
        df, horizon=horizon, flat_threshold=flat_threshold, binary=binary
    )
    return df


def merge_news_features(
    panel: pd.DataFrame,
    news_df: pd.DataFrame,
    *,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Join same-day news features by (date, ticker). No lookahead."""
    if news_df.empty:
        out = panel.copy()
        for col in NEWS_FEATURE_COLUMNS:
            out[col] = fill_value
        return out
    nf = news_df.copy()
    nf["date"] = pd.to_datetime(nf["date"], errors="coerce").dt.date
    out = panel.copy()
    out["_date"] = pd.to_datetime(out.index).date
    ticker_col = "Ticker" if "Ticker" in out.columns else None
    if ticker_col is None:
        for col in NEWS_FEATURE_COLUMNS:
            out[col] = fill_value
        return out.drop(columns=["_date"], errors="ignore")
    merged = out.reset_index().merge(
        nf[["date", "ticker"] + [c for c in NEWS_FEATURE_COLUMNS if c in nf.columns]],
        left_on=["_date", ticker_col],
        right_on=["date", "ticker"],
        how="left",
    )
    for col in NEWS_FEATURE_COLUMNS:
        if col in merged.columns:
            merged[col] = merged[col].fillna(fill_value)
        else:
            merged[col] = fill_value
    idx_col = merged.columns[0]
    merged = merged.set_index(idx_col)
    drop_cols = [c for c in ("date", "ticker", "_date") if c in merged.columns]
    return merged.drop(columns=drop_cols, errors="ignore")


def make_sample_weights(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    news_boost: float = 3.0,
    directional_boost: float = 1.5,
) -> np.ndarray:
    """Class-balanced weights; boost non-flat labels and rows with real news."""
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    class_w = {
        c: float(len(y_arr)) / (len(classes) * float(cnt)) for c, cnt in zip(classes, counts)
    }
    w = np.array([class_w[yi] for yi in y_arr], dtype=float)
    w = np.where(y_arr != 0, w * directional_boost, w)
    news_cols = [c for c in NEWS_FEATURE_COLUMNS if c in X.columns]
    if news_cols and news_boost > 1.0:
        has_news = (X[news_cols].abs().sum(axis=1).to_numpy() > 0)
        w = np.where(has_news, w * news_boost, w)
    return w


def compute_feature_importance(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    max_samples: int = 2000,
) -> dict[str, float]:
    """Permutation importance on a sample of the test set."""
    from sklearn.inspection import permutation_importance

    if X.empty or len(y) == 0:
        return {}
    if len(X) > max_samples:
        X = X.iloc[-max_samples:]
        y = y.iloc[-max_samples:]
    try:
        result = permutation_importance(
            model, X, y, n_repeats=5, random_state=42, n_jobs=1
        )
        return {
            str(col): float(imp)
            for col, imp in zip(X.columns, result.importances_mean)
        }
    except Exception:  # noqa: BLE001
        return {}


def _build_labeled_panel(
    frames: dict[str, pd.DataFrame],
    *,
    horizon: int,
    flat_threshold: float,
    news_df: pd.DataFrame | None,
    use_news_features: bool,
    spy_close: pd.Series | None = None,
    binary: bool = False,
) -> pd.DataFrame:
    spy = spy_close
    if spy is None and "SPY" in frames:
        spy = frames["SPY"]["Close"]
    panels: list[pd.DataFrame] = []
    for ticker, ohlcv in frames.items():
        panel = prepare_training_frame(
            ohlcv,
            horizon=horizon,
            flat_threshold=flat_threshold,
            spy_close=spy,
            binary=binary,
        )
        panel["Ticker"] = ticker
        if use_news_features:
            panel = merge_news_features(panel, news_df if news_df is not None else pd.DataFrame())
        panels.append(panel)
    return pd.concat(panels, axis=0).sort_index()


def _apply_lookback(
    data: pd.DataFrame,
    *,
    use_news_features: bool,
    news_df: pd.DataFrame | None,
    news_window_only: bool,
    price_lookback_days: int | None,
) -> pd.DataFrame:
    idx = pd.to_datetime(data.index)
    if use_news_features and news_df is not None and not news_df.empty and news_window_only:
        nd = pd.to_datetime(news_df["date"], errors="coerce")
        n_start, n_end = nd.min(), nd.max()
        if pd.notna(n_start) and pd.notna(n_end):
            return data.loc[(idx >= n_start) & (idx <= n_end)]
    if price_lookback_days:
        cutoff = idx.max() - pd.Timedelta(days=int(price_lookback_days))
        return data.loc[idx >= cutoff]
    return data


def _score_split(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[float, float, np.ndarray]:
    if len(X) == 0:
        return float("nan"), float("nan"), np.array([])
    pred = model.predict(X)
    return float(accuracy_score(y, pred)), directional_hit_rate(y, pred), pred


def train_directional_model(
    frames: dict[str, pd.DataFrame],
    horizon: int = 5,
    flat_threshold: float = 0.005,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    news_df: pd.DataFrame | None = None,
    use_news_features: bool = False,
    news_window_only: bool = False,
    price_lookback_days: int | None = None,
    news_boost: float = 3.0,
    directional_boost: float = 1.5,
    model_params: dict[str, Any] | None = None,
    refit_train_val: bool = False,
    spy_close: pd.Series | None = None,
    binary: bool = False,
) -> tuple[Pipeline, TrainResult, pd.DataFrame]:
    """Train on concatenated multi-ticker chronological panels."""
    feature_cols = ALL_FEATURE_COLUMNS if use_news_features else FEATURE_COLUMNS
    data = _build_labeled_panel(
        frames,
        horizon=horizon,
        flat_threshold=flat_threshold,
        news_df=news_df,
        use_news_features=use_news_features,
        spy_close=spy_close,
        binary=binary,
    )
    data = _apply_lookback(
        data,
        use_news_features=use_news_features,
        news_df=news_df,
        news_window_only=news_window_only,
        price_lookback_days=price_lookback_days,
    )
    if price_lookback_days:
        print(
            f"[train] lookback={price_lookback_days}d use_news={use_news_features} "
            f"flat={flat_threshold} horizon={horizon} binary={binary} ({len(data)} rows)"
        )

    X, y = feature_matrix(data, feature_cols=feature_cols)
    clean = data.dropna(
        subset=[c for c in feature_cols + ["direction", "forward_return"] if c in data.columns]
    )

    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(
        X, y, train_ratio=train_ratio, val_ratio=val_ratio
    )
    params = dict(model_params or {})
    params.setdefault("use_news_features", use_news_features)
    model = build_model(**params)
    sw = make_sample_weights(
        X_train, y_train, news_boost=news_boost, directional_boost=directional_boost
    )
    model.fit(X_train, y_train, clf__sample_weight=sw)

    val_acc, val_hit, _ = _score_split(model, X_val, y_val)
    test_acc, test_hit, pred_test = _score_split(model, X_test, y_test)

    if refit_train_val and len(X_val):
        X_tv = pd.concat([X_train, X_val])
        y_tv = pd.concat([y_train, y_val])
        sw_tv = make_sample_weights(
            X_tv, y_tv, news_boost=news_boost, directional_boost=directional_boost
        )
        model = build_model(**params)
        model.fit(X_tv, y_tv, clf__sample_weight=sw_tv)

    fi = compute_feature_importance(model, X_test, y_test) if len(X_test) else {}

    result = TrainResult(
        train_accuracy=float(accuracy_score(y_train, model.predict(X_train))),
        val_accuracy=val_acc,
        test_accuracy=test_acc,
        hit_rate_directional=test_hit,
        val_hit_rate_directional=val_hit,
        n_train=len(X_train),
        n_val=len(X_val),
        n_test=len(X_test),
        feature_columns=list(X.columns),
        report=classification_report(y_test, pred_test, zero_division=0) if len(X_test) else "",
        model_name="HistGradientBoostingClassifier",
        feature_importance=fi or None,
        best_params={
            "use_news_features": use_news_features,
            "flat_threshold": flat_threshold,
            "horizon": horizon,
            "binary": binary,
            "price_lookback_days": price_lookback_days,
            "news_boost": news_boost,
            "directional_boost": directional_boost,
            **{k: v for k, v in params.items() if k != "use_news_features"},
        },
    )
    return model, result, clean


def _candidate_grid(news_available: bool, *, system: bool = False) -> list[dict[str, Any]]:
    """Curated search space — scored on validation directional hit − overfit penalty."""
    hgb_space = [
        {"max_depth": 3, "learning_rate": 0.05, "min_samples_leaf": 40, "l2_regularization": 0.5, "max_iter": 250},
        {"max_depth": 4, "learning_rate": 0.05, "min_samples_leaf": 30, "l2_regularization": 0.3, "max_iter": 300},
        {"max_depth": 5, "learning_rate": 0.03, "min_samples_leaf": 20, "l2_regularization": 0.1, "max_iter": 350},
    ]
    if system:
        data_space: list[dict[str, Any]] = []
        for horizon in (3, 5, 10):
            for lookback in (None, 504):
                for binary in (False, True):
                    data_space.append(
                        {
                            "use_news_features": False,
                            "price_lookback_days": lookback,
                            "flat_threshold": 0.008,
                            "horizon": horizon,
                            "binary": binary,
                            "news_boost": 1.0,
                        }
                    )
        hgb_compact = [hgb_space[0], hgb_space[1]]
    else:
        data_space = [
            {"use_news_features": False, "price_lookback_days": 252, "flat_threshold": 0.008, "news_boost": 1.0, "horizon": 5, "binary": False},
            {"use_news_features": False, "price_lookback_days": 504, "flat_threshold": 0.008, "news_boost": 1.0, "horizon": 5, "binary": False},
            {"use_news_features": False, "price_lookback_days": 504, "flat_threshold": 0.012, "news_boost": 1.0, "horizon": 5, "binary": False},
            {"use_news_features": False, "price_lookback_days": None, "flat_threshold": 0.008, "news_boost": 1.0, "horizon": 5, "binary": False},
        ]
        if news_available:
            data_space.extend(
                [
                    {"use_news_features": True, "price_lookback_days": 504, "flat_threshold": 0.008, "news_boost": 3.0, "horizon": 5, "binary": False},
                    {"use_news_features": True, "price_lookback_days": None, "flat_threshold": 0.008, "news_boost": 3.0, "horizon": 5, "binary": False},
                ]
            )
        hgb_compact = hgb_space

    candidates: list[dict[str, Any]] = []
    for data_cfg in data_space:
        for hgb in hgb_compact:
            candidates.append({**data_cfg, "model_params": hgb, "directional_boost": 1.5})
    return candidates


def optimize_directional_model(
    frames: dict[str, pd.DataFrame],
    *,
    news_df: pd.DataFrame | None = None,
    horizon: int = 5,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    verbose: bool = True,
    system: bool = False,
) -> tuple[Pipeline, TrainResult, pd.DataFrame]:
    """Pick best config by validation directional hit; report held-out test once."""
    news_available = news_df is not None and not news_df.empty
    candidates = _candidate_grid(news_available, system=system)
    trials: list[dict[str, Any]] = []
    best: tuple[tuple[float, float, float], dict[str, Any], Pipeline, TrainResult, pd.DataFrame] | None = None
    spy = frames["SPY"]["Close"] if "SPY" in frames else None

    for i, cand in enumerate(candidates, start=1):
        use_news = bool(cand["use_news_features"])
        if use_news and not news_available:
            continue
        cand_horizon = int(cand.get("horizon", horizon))
        cand_binary = bool(cand.get("binary", False))
        try:
            model, result, clean = train_directional_model(
                frames,
                horizon=cand_horizon,
                flat_threshold=float(cand["flat_threshold"]),
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                news_df=news_df,
                use_news_features=use_news,
                news_window_only=False,
                price_lookback_days=cand.get("price_lookback_days"),
                news_boost=float(cand.get("news_boost", 1.0)),
                directional_boost=float(cand.get("directional_boost", 1.5)),
                model_params=cand.get("model_params"),
                refit_train_val=False,
                spy_close=spy,
                binary=cand_binary,
            )
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[optimize] trial {i}/{len(candidates)} failed: {exc}")
            continue

        trial = {
            "trial": i,
            "use_news": use_news,
            "lookback": cand.get("price_lookback_days"),
            "flat_threshold": cand["flat_threshold"],
            "horizon": cand_horizon,
            "binary": cand_binary,
            "news_boost": cand.get("news_boost"),
            "model_params": cand.get("model_params"),
            "val_hit": result.val_hit_rate_directional,
            "val_acc": result.val_accuracy,
            "train_acc": result.train_accuracy,
            "test_hit": result.hit_rate_directional,
            "test_acc": result.test_accuracy,
            "n_train": result.n_train,
            "overfit_gap": float(result.train_accuracy - result.val_accuracy),
        }
        overfit_gap = max(0.0, float(result.train_accuracy - result.val_accuracy))
        score = (
            float(result.val_hit_rate_directional) - 0.4 * overfit_gap,
            float(result.val_accuracy) - 0.2 * overfit_gap,
            float(result.n_train),
        )
        trials.append(trial)
        if verbose:
            print(
                f"[optimize] {i}/{len(candidates)} news={use_news} "
                f"h={cand_horizon} bin={cand_binary} "
                f"lb={cand.get('price_lookback_days')} flat={cand['flat_threshold']} "
                f"val_hit={result.val_hit_rate_directional:.3f} "
                f"gap={overfit_gap:.3f} score={score[0]:.3f} "
                f"test_hit={result.hit_rate_directional:.3f}"
            )

        if best is None or score > best[0]:
            best = (score, cand, model, result, clean)

    if best is None:
        raise RuntimeError("Optimization produced no successful trials")

    _score, cand, _model, pre_result, _clean = best
    cand_horizon = int(cand.get("horizon", horizon))
    cand_binary = bool(cand.get("binary", False))
    model, result, clean = train_directional_model(
        frames,
        horizon=cand_horizon,
        flat_threshold=float(cand["flat_threshold"]),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        news_df=news_df,
        use_news_features=bool(cand["use_news_features"]),
        news_window_only=False,
        price_lookback_days=cand.get("price_lookback_days"),
        news_boost=float(cand.get("news_boost", 1.0)),
        directional_boost=float(cand.get("directional_boost", 1.5)),
        model_params=cand.get("model_params"),
        refit_train_val=True,
        spy_close=spy,
        binary=cand_binary,
    )
    result.test_accuracy = pre_result.test_accuracy
    result.hit_rate_directional = pre_result.hit_rate_directional
    result.report = pre_result.report
    result.feature_importance = pre_result.feature_importance
    result.val_accuracy = pre_result.val_accuracy
    result.val_hit_rate_directional = pre_result.val_hit_rate_directional
    result.n_test = pre_result.n_test
    result.optimization_trials = sorted(
        trials,
        key=lambda t: (t["val_hit"] - 0.4 * t.get("overfit_gap", 0.0), t["val_acc"]),
        reverse=True,
    )
    result.best_params = {
        "use_news_features": bool(cand["use_news_features"]),
        "flat_threshold": cand["flat_threshold"],
        "horizon": cand_horizon,
        "binary": cand_binary,
        "price_lookback_days": cand.get("price_lookback_days"),
        "news_boost": cand.get("news_boost"),
        "directional_boost": cand.get("directional_boost"),
        **(cand.get("model_params") or {}),
        "selection_metric": "val_directional_hit_minus_overfit_penalty",
        "n_trials": len(trials),
    }
    if verbose:
        print(
            f"[optimize] WINNER news={cand['use_news_features']} "
            f"h={cand_horizon} bin={cand_binary} "
            f"lb={cand.get('price_lookback_days')} flat={cand['flat_threshold']} "
            f"val_hit={result.val_hit_rate_directional:.3f} "
            f"test_hit={result.hit_rate_directional:.3f} "
            f"test_acc={result.test_accuracy:.3f}"
        )
    return model, result, clean



def train_ab_comparison(
    frames: dict[str, pd.DataFrame],
    settings: Any | None = None,
    **train_kwargs: Any,
) -> dict[str, Any]:
    """A/B: technical-only vs technical+news on the same lookback."""
    news_df = load_news_features_table(settings) if settings is not None else pd.DataFrame()
    lookback = train_kwargs.pop("price_lookback_days", 504)
    _, result_a, _ = train_directional_model(
        frames,
        use_news_features=False,
        news_window_only=False,
        price_lookback_days=lookback,
        **train_kwargs,
    )
    result_b = None
    if not news_df.empty:
        _, result_b, _ = train_directional_model(
            frames,
            use_news_features=True,
            news_df=news_df,
            news_window_only=False,
            price_lookback_days=lookback,
            **train_kwargs,
        )
    return {
        "technical_only": {
            "test_accuracy": result_a.test_accuracy,
            "hit_rate_directional": result_a.hit_rate_directional,
            "val_hit_rate_directional": result_a.val_hit_rate_directional,
            "n_test": result_a.n_test,
        },
        "technical_plus_news": (
            {
                "test_accuracy": result_b.test_accuracy,
                "hit_rate_directional": result_b.hit_rate_directional,
                "val_hit_rate_directional": result_b.val_hit_rate_directional,
                "n_test": result_b.n_test,
                "news_rows": len(news_df),
            }
            if result_b
            else None
        ),
        "news_data_available": not news_df.empty,
        "lookback_days": lookback,
        "lift_directional_pp": (
            (result_b.hit_rate_directional - result_a.hit_rate_directional) * 100
            if result_b is not None
            and not np.isnan(result_a.hit_rate_directional)
            and not np.isnan(result_b.hit_rate_directional)
            else None
        ),
    }


def save_model(model: Pipeline, meta: dict[str, Any], artifact_dir: str | Path) -> Path:
    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    model_path = path / "directional_model.joblib"
    meta_path = path / "model_meta.json"
    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return model_path


def load_model(artifact_dir: str | Path) -> tuple[Pipeline, dict[str, Any]]:
    path = Path(artifact_dir)
    model = joblib.load(path / "directional_model.joblib")
    meta = json.loads((path / "model_meta.json").read_text(encoding="utf-8"))
    return model, meta


def predict_proba_direction(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return class probability matrix aligned to model.classes_."""
    return model.predict_proba(X)


def latest_feature_row(
    ohlcv: pd.DataFrame,
    *,
    news_features: dict[str, float] | None = None,
    feature_columns: list[str] | None = None,
    spy_close: pd.Series | None = None,
) -> pd.DataFrame:
    """Latest bar features; optionally append Tiingo/news columns to match trained model."""
    cols = list(feature_columns or FEATURE_COLUMNS)
    tech_cols = [c for c in cols if c in FEATURE_COLUMNS]
    df = add_indicators(ohlcv, spy_close=spy_close)
    row = df.dropna(subset=tech_cols).iloc[[-1]][tech_cols].copy()
    for c in cols:
        if c in NEWS_FEATURE_COLUMNS:
            val = 0.0
            if news_features and c in news_features:
                val = float(news_features[c])
            row[c] = val
    return row[cols]
