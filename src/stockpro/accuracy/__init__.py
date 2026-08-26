"""Realized model accuracy: log predictions, resolve after forward horizon."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from stockpro.config import Settings
from stockpro.data import download_ohlcv
from stockpro.signals import Signal


PREDICTION_COLUMNS = [
    "as_of",
    "ticker",
    "signal",
    "predicted_class",
    "confidence",
    "spot",
    "horizon_days",
    "resolve_on",
    "actual_return",
    "actual_class",
    "correct",
    "resolved",
    "logged_at",
]


def _predictions_path(settings: Settings) -> Path:
    journal_dir = Path((settings.get("journal", default={}) or {}).get("directory", "data/journal"))
    path = journal_dir / "predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _direction_from_return(ret: float, flat_threshold: float) -> int:
    if ret > flat_threshold:
        return 1
    if ret < -flat_threshold:
        return -1
    return 0


def _is_resolved(val: Any) -> bool:
    return str(val).strip().lower() in {"true", "1", "yes"}


def _session_dates(ohlcv: pd.DataFrame) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(ohlcv.index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.normalize()


def log_prediction(
    settings: Settings,
    *,
    as_of: date | str,
    ticker: str,
    signal: Signal | str,
    predicted_class: int,
    confidence: float,
    spot: float,
    horizon_days: int | None = None,
) -> None:
    """Append one model prediction to resolve later."""
    feat = settings.get("features", default={}) or {}
    horizon = int(horizon_days or feat.get("forward_horizon_days", 5))
    as_of_d = date.fromisoformat(str(as_of)[:10])
    # Matches features.add_forward_labels: need close at as_of + horizon sessions.
    # Resolve once that session's bar can appear in Yahoo (typically next calendar morning).
    target_session = (pd.Timestamp(as_of_d) + pd.tseries.offsets.BDay(horizon)).date()
    resolve_on = target_session + timedelta(days=1)

    row = {
        "as_of": as_of_d.isoformat(),
        "ticker": ticker,
        "signal": signal.value if isinstance(signal, Signal) else str(signal),
        "predicted_class": int(predicted_class),
        "confidence": float(confidence),
        "spot": float(spot),
        "horizon_days": horizon,
        "resolve_on": resolve_on.isoformat(),
        "actual_return": "",
        "actual_class": "",
        "correct": "",
        "resolved": False,
        "logged_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    path = _predictions_path(settings)
    df = pd.DataFrame([row], columns=PREDICTION_COLUMNS)
    if path.exists():
        prev = pd.read_csv(path)
        for c in PREDICTION_COLUMNS:
            if c not in prev.columns:
                prev[c] = ""
        # Avoid duplicate same-day ticker rows (re-runs)
        prev = prev[
            ~(
                (prev["as_of"].astype(str) == row["as_of"])
                & (prev["ticker"].astype(str) == row["ticker"])
            )
        ]
        df = pd.concat([prev[PREDICTION_COLUMNS], df], ignore_index=True)
    df.to_csv(path, index=False)


def resolve_predictions(settings: Settings, as_of: date | None = None) -> dict[str, Any]:
    """Fill actual forward returns for due predictions; return rolling accuracy stats.

    Label definition matches ``features.add_forward_labels``:
    ``Close[t+horizon] / Close[t] - 1``.
    """
    as_of = as_of or date.today()
    path = _predictions_path(settings)
    if not path.exists():
        return {
            "resolved_today": 0,
            "n_resolved": 0,
            "accuracy_3class": None,
            "accuracy_directional": None,
            "n_directional": 0,
        }

    df = pd.read_csv(path)
    for c in PREDICTION_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    feat = settings.get("features", default={}) or {}
    flat_threshold = float(feat.get("flat_threshold", 0.005))
    uni = settings.get("universe", default={}) or {}
    history_start = uni.get("history_start", "2000-01-01")

    resolved_today = 0
    cache: dict[str, pd.DataFrame] = {}

    for i, row in df.iterrows():
        if _is_resolved(row.get("resolved")):
            continue

        ticker = str(row["ticker"])
        pred_day = date.fromisoformat(str(row["as_of"])[:10])
        horizon = int(row.get("horizon_days") or feat.get("forward_horizon_days", 5))
        # Forward bar date required by training labels (Close[t+horizon]).
        target_session = (pd.Timestamp(pred_day) + pd.tseries.offsets.BDay(horizon)).date()
        if as_of < target_session:
            continue

        if ticker not in cache:
            try:
                # Yahoo ``end`` is exclusive; pad so the latest session is included.
                end = datetime.combine(as_of + timedelta(days=2), datetime.min.time(), tzinfo=timezone.utc)
                cache[ticker] = download_ohlcv(ticker, start=history_start, end=end)
            except Exception as exc:  # noqa: BLE001
                print(f"[accuracy] skip resolve {ticker}: {exc}")
                continue

        ohlcv = cache[ticker]
        sessions = _session_dates(ohlcv)
        # First session on/after prediction day (matches training bar alignment)
        hits = sessions[sessions.date >= pred_day]
        if len(hits) == 0:
            continue
        start_ts = hits[0]
        start_i = int(sessions.get_indexer([start_ts])[0])
        if start_i < 0:
            continue
        end_i = start_i + horizon
        if end_i >= len(ohlcv):
            continue

        spot0 = float(ohlcv.iloc[start_i]["Close"])
        spot1 = float(ohlcv.iloc[end_i]["Close"])
        if spot0 <= 0:
            continue
        actual_ret = spot1 / spot0 - 1.0
        actual_class = _direction_from_return(actual_ret, flat_threshold)
        predicted = int(row["predicted_class"])
        correct = int(predicted == actual_class)

        df.at[i, "actual_return"] = round(actual_ret, 6)
        df.at[i, "actual_class"] = actual_class
        df.at[i, "correct"] = correct
        df.at[i, "resolved"] = True
        resolved_today += 1

    df.to_csv(path, index=False)

    done = df[df["resolved"].map(_is_resolved)].copy()
    if done.empty:
        return {
            "resolved_today": resolved_today,
            "n_resolved": 0,
            "accuracy_3class": None,
            "accuracy_directional": None,
            "n_directional": 0,
        }

    correct = pd.to_numeric(done["correct"], errors="coerce").fillna(0)
    pred = pd.to_numeric(done["predicted_class"], errors="coerce")
    actual = pd.to_numeric(done["actual_class"], errors="coerce")
    acc = float(correct.mean())

    mask = pred != 0
    if mask.any():
        dir_acc = float((pred[mask] == actual[mask]).mean())
        n_dir = int(mask.sum())
    else:
        dir_acc = None
        n_dir = 0

    conf = pd.to_numeric(done["confidence"], errors="coerce")
    thr = float((settings.get("model", default={}) or {}).get("probability_threshold", 0.65))
    high = conf >= thr
    high_acc = float(correct[high].mean()) if high.any() else None

    by_ticker: dict[str, float] = {}
    for tkr, g in done.groupby(done["ticker"].astype(str)):
        c = pd.to_numeric(g["correct"], errors="coerce").fillna(0)
        if len(c):
            by_ticker[str(tkr)] = float(c.mean())

    return {
        "resolved_today": resolved_today,
        "n_resolved": int(len(done)),
        "accuracy_3class": acc,
        "accuracy_directional": dir_acc,
        "n_directional": n_dir,
        "accuracy_high_confidence": high_acc,
        "n_high_confidence": int(high.sum()) if high.any() else 0,
        "accuracy_by_ticker": by_ticker,
    }
