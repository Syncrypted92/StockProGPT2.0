"""Map model scores to bullish / bearish / flat signals."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class Signal(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    FLAT = "flat"


@dataclass
class SignalResult:
    ticker: str
    signal: Signal
    predicted_class: int
    confidence: float
    probabilities: dict[str, float]


def score_to_signal(
    model: Pipeline,
    features: pd.DataFrame,
    ticker: str,
    probability_threshold: float = 0.55,
) -> SignalResult:
    classes = list(model.named_steps["clf"].classes_)
    proba = model.predict_proba(features)[0]
    pred = int(model.predict(features)[0])
    class_to_p = {int(c): float(p) for c, p in zip(classes, proba)}
    confidence = class_to_p.get(pred, float(np.max(proba)))

    if pred == 1 and confidence >= probability_threshold:
        signal = Signal.BULLISH
    elif pred == -1 and confidence >= probability_threshold:
        signal = Signal.BEARISH
    else:
        signal = Signal.FLAT

    return SignalResult(
        ticker=ticker,
        signal=signal,
        predicted_class=pred,
        confidence=confidence,
        probabilities={str(k): v for k, v in class_to_p.items()},
    )
