"""
app.py — FastAPI surface for the daily check-in.

Flow per check-in:
  1. user submits mood tap (+ optional free text)
  2. emotion model -> distress; crisis screen runs FIRST (safety before analytics)
  3. compose wellbeing index, update the user's trend
  4. build the response: gentle suggestion + (if needed) crisis resources/handoff
  5. write a minimal audit record if any risk tier fired

This is a reference skeleton. Production needs: real auth, encrypted storage of
mental-health data (special category), per-user keys, data-retention controls,
a staffed safety queue if you offer human review, and a privacy/consent layer.
Talk to a clinician and a lawyer before launch — this is not legal advice.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import datetime as dt

# project imports
from ml.emotion import EmotionModel
from ml.trends import TrendTracker
from core.scoring import compose_wellbeing
from core.crisis import (assess, plan_response, make_audit, ContactConsent,
                         RiskTier)
from config.resources import for_country

app = FastAPI(title="Mindful check-in (reference)")
_emotion = EmotionModel()

# In-memory stores stand in for a database. DO NOT ship this; mental-health data
# must be encrypted at rest with proper retention controls.
_trends: dict[str, TrendTracker] = {}
_consents: dict[str, ContactConsent] = {}
_audit_log: list = []


class CheckIn(BaseModel):
    user_id: str
    mood: int                      # 1..5 tap
    text: Optional[str] = None
    country: Optional[str] = "IN"
    phq9: Optional[int] = None
    gad7: Optional[int] = None


@app.post("/checkin")
def checkin(c: CheckIn):
    # 2. emotion + crisis FIRST
    emo = _emotion.predict(c.text or "")
    crisis = assess(c.text or "", model_risk=0.0)  # plug a risk model here if you have one
    distress_t1 = emo.distress >= 0.6
    action = plan_response(crisis, _consents.get(c.user_id), distress_tier_1=distress_t1)

    # 3. wellbeing + trend
    wellbeing = compose_wellbeing(c.mood, emo.distress, c.phq9, c.gad7)
    tracker = _trends.setdefault(c.user_id, TrendTracker())
    signal = tracker.observe(wellbeing)

    # 5. audit if any risk tier fired
    if crisis.tier > RiskTier.NONE or action.show_resources:
        _audit_log.append(make_audit(c.user_id, crisis, action, c.text or ""))

    return {
        "date": dt.date.today().isoformat(),
        "emotion": {"dominant": emo.dominant, "distress": round(emo.distress, 3),
                    "buckets": {k: round(v, 3) for k, v in emo.emotions.items()},
                    "model_fallback": emo.used_fallback},
        "wellbeing": round(wellbeing, 3),
        "trend": {"ewma": round(signal.ewma, 3),
                  "robust_z": None if signal.robust_z is None else round(signal.robust_z, 2),
                  "severity": signal.severity,
                  "sustained_drop": signal.sustained_drop},
        "support": {
            "show_resources": action.show_resources,
            "prominent": action.prominent,
            "warm_handoff": action.offer_warm_handoff,
            "message": action.message,
            "resources": [r.__dict__ for r in for_country(c.country)]
                          if action.show_resources else [],
            # The app must render this proposal as a cancellable prompt, NOT send silently.
            "contact_alert_proposed": action.propose_contact_alert is not None,
        },
        "suggestion": _suggestion(signal.severity, emo.dominant),
    }


def _suggestion(severity: str, dominant: str) -> str:
    """Gentle, non-prescriptive. Suggestions, never directives; nothing that
    doubles as a coping mechanism through discomfort."""
    if severity in ("notable", "marked"):
        return ("Your recent check-ins look tougher than your usual. That's worth "
                "being kind to yourself about — and a counsellor or someone you "
                "trust can help carry it.")
    if dominant == "fear":
        return "If your mind is racing, a few slow breaths or a short walk can take the edge off."
    if dominant == "sadness":
        return "Low days are allowed. A small, kind thing for yourself counts."
    return "Thanks for checking in. Noticing how you feel is a real step."


@app.post("/consent/contact")
def set_contact_consent(user_id: str, contact_label: str, channel: str = "sms"):
    """User explicitly pre-authorizes a trusted-contact alert at EXPLICIT tier.
    Revoke with DELETE. This is the ONLY way a third-party alert can ever fire."""
    _consents[user_id] = ContactConsent(contact_label, channel)
    return {"ok": True, "consent": _consents[user_id].__dict__}


@app.delete("/consent/contact")
def revoke_contact_consent(user_id: str):
    _consents.pop(user_id, None)
    return {"ok": True}
