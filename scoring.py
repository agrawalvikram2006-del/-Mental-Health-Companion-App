"""
scoring.py — compose a single wellbeing index from multiple signals, explicitly.

Keep this transparent. A clinician or auditor should be able to read exactly how
a number was produced. No hidden model here on purpose.

Inputs (all optional except self_report_mood):
  self_report_mood : 1-5 the user taps ("how are you today?")     -> primary
  text_distress    : [0,1] from ml/emotion.py                      -> secondary
  phq9 / gad7      : periodic validated scales (see below)         -> anchors

Output: wellbeing in [0,1], higher = better.
"""

from __future__ import annotations
from typing import Optional


def compose_wellbeing(self_report_mood: int,
                      text_distress: Optional[float] = None,
                      phq9: Optional[int] = None,
                      gad7: Optional[int] = None) -> float:
    # Normalize self-report 1..5 -> 0..1
    mood = (max(1, min(5, self_report_mood)) - 1) / 4.0

    parts = [(mood, 0.6)]
    if text_distress is not None:
        parts.append((1.0 - max(0.0, min(1.0, text_distress)), 0.25))
    if phq9 is not None:
        # PHQ-9 range 0..27 (higher = more depressive symptoms)
        parts.append((1.0 - min(27, max(0, phq9)) / 27.0, 0.1))
    if gad7 is not None:
        # GAD-7 range 0..21 (higher = more anxiety)
        parts.append((1.0 - min(21, max(0, gad7)) / 21.0, 0.05))

    wsum = sum(w for _, w in parts)
    return sum(v * w for v, w in parts) / wsum


# Validated instruments belong in the product, administered on a cadence, with
# their standard scoring and severity bands. These are screening tools, not
# diagnoses. PHQ-9 item 9 in particular asks about self-harm and MUST route into
# the crisis flow regardless of the total score.
PHQ9_SEVERITY = [(0, 4, "minimal"), (5, 9, "mild"), (10, 14, "moderate"),
                 (15, 19, "moderately severe"), (20, 27, "severe")]
GAD7_SEVERITY = [(0, 4, "minimal"), (5, 9, "mild"), (10, 14, "moderate"),
                 (15, 21, "severe")]


def band(score: int, table) -> str:
    for lo, hi, label in table:
        if lo <= score <= hi:
            return label
    return "unknown"
