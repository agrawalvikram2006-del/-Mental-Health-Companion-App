"""
trends.py — per-user baseline learning and trend / deviation detection.

DESIGN NOTE — why robust stats are the primary method, not an LSTM:
Daily check-ins are sparse, noisy, and unevenly spaced. A deep model trained
per-user overfits badly and is hard to explain to a clinician or to the user.
So the *primary* deviation signal here is:

  * a robust personal baseline (median + MAD, resistant to outliers),
  * an EWMA to track the recent trajectory,
  * a robust z-score for "how far is today from this person's normal",
  * a CUSUM-style changepoint flag for sustained shifts (a slow slide into a
    low mood matters more than one bad day).

The LSTM forecaster is provided as an OPTIONAL extra for users with long, dense
histories. It forecasts; it does not gate any alert on its own.

Everything here operates on a "wellbeing index" in [0,1] (higher = better),
which you compose from: 1 - distress (from emotion.py), self-reported mood,
and optionally periodic PHQ-9/GAD-7 scores. Keep the composition explicit and
auditable in core/scoring.py rather than hiding it in a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Optional
import statistics
import math


@dataclass
class Baseline:
    median: float
    mad: float          # median absolute deviation (robust spread)
    n: int


@dataclass
class DeviationSignal:
    value: float                # today's wellbeing index
    baseline: Optional[Baseline]
    robust_z: Optional[float]   # negative = worse than personal normal
    ewma: float
    sustained_drop: bool        # changepoint flag
    severity: str               # "none" | "mild" | "notable" | "marked"


_MAD_TO_STD = 1.4826  # makes MAD comparable to a standard deviation for normal data


class TrendTracker:
    """Online, per-user. Persist `history` (a list of floats) between sessions."""

    def __init__(self, history: Optional[list[float]] = None,
                 ewma_alpha: float = 0.3, min_baseline: int = 7,
                 cusum_k: float = 0.5, cusum_h: float = 4.0):
        self.history: list[float] = list(history or [])
        self.alpha = ewma_alpha
        self.min_baseline = min_baseline
        self.cusum_k = cusum_k          # slack
        self.cusum_h = cusum_h          # decision threshold
        self._cusum_neg = 0.0
        self._ewma: Optional[float] = None
        # warm up internal state from history
        for v in self.history:
            self._update_ewma(v)

    # -- baseline over the trailing window, excluding the most recent point ----
    def _baseline(self, exclude_last: bool = True) -> Optional[Baseline]:
        data = self.history[:-1] if exclude_last and self.history else self.history
        if len(data) < self.min_baseline:
            return None
        med = statistics.median(data)
        raw_mad = statistics.median([abs(x - med) for x in data])
        # Floor the spread. On a 0-1 wellbeing scale, a personal "normal" is never
        # truly zero-variance; without this, one ordinary dip yields an absurd
        # z-score. 0.05 ≈ a quarter of one mood-tap step.
        mad = max(raw_mad, 0.05)
        return Baseline(med, mad, len(data))

    def _update_ewma(self, v: float):
        self._ewma = v if self._ewma is None else self.alpha * v + (1 - self.alpha) * self._ewma

    def observe(self, wellbeing: float) -> DeviationSignal:
        """Call once per check-in with the composed wellbeing index in [0,1]."""
        wellbeing = max(0.0, min(1.0, wellbeing))
        base = self._baseline(exclude_last=False)  # baseline from prior points
        self.history.append(wellbeing)
        self._update_ewma(wellbeing)

        robust_z = None
        if base:
            robust_z = (wellbeing - base.median) / (base.mad * _MAD_TO_STD)

        # CUSUM on the downside only (we care about sustained worsening).
        if base:
            standardized = (wellbeing - base.median) / (base.mad * _MAD_TO_STD)
            self._cusum_neg = min(0.0, self._cusum_neg + standardized + self.cusum_k)
        sustained = self._cusum_neg < -self.cusum_h
        if sustained:
            self._cusum_neg = 0.0  # reset after flagging

        severity = self._severity(robust_z, sustained)
        return DeviationSignal(
            value=wellbeing, baseline=base, robust_z=robust_z,
            ewma=self._ewma if self._ewma is not None else wellbeing,
            sustained_drop=sustained, severity=severity,
        )

    @staticmethod
    def _severity(z: Optional[float], sustained: bool) -> str:
        if z is None:
            return "none"
        if z <= -3.0 or sustained:
            return "marked"
        if z <= -2.0:
            return "notable"
        if z <= -1.5:
            return "mild"
        return "none"


# ---------------------------------------------------------------------------
# OPTIONAL: tiny LSTM forecaster for users with long, dense histories.
# Forecast only; never the sole basis for an alert.
# ---------------------------------------------------------------------------
def build_lstm(seq_len: int = 14):
    """Returns a small Keras LSTM that predicts the next wellbeing value."""
    from tensorflow import keras
    from tensorflow.keras import layers
    model = keras.Sequential([
        layers.Input((seq_len, 1)),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),  # wellbeing in [0,1]
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def make_windows(series: list[float], seq_len: int = 14):
    import numpy as np
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])
    X = np.array(X).reshape(-1, seq_len, 1)
    return X, np.array(y)
