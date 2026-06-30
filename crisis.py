"""
crisis.py — conservative, layered crisis screening.

PHILOSOPHY
----------
Two different decisions, two different error tolerances:

  A) "Should we show this user crisis resources right now?"
     -> Favor SENSITIVITY. Showing a helpline to someone who didn't need it
        costs little. Missing someone who did costs everything. Over-trigger.

  B) "Should we tell another human being about this user's state?"
     -> Favor SPECIFICITY + CONSENT + HUMAN REVIEW. Disclosing someone's mental
        state to a third party without their informed, revocable consent is a
        privacy violation and can break the trust that keeps them using the app
        at all. Never silent, never automatic, never ML-only.

This module NEVER triggers automated emergency/police dispatch. Coordinated
in-person response, when truly needed, is handled by trained crisis lines
(988 / Tele-MANAS), which escalate conservatively. An app auto-dispatching on a
classifier output endangers users.

DETECTION = high-recall phrase layer  OR/combined-with  ML classifier.
The phrase layer exists because ML models miss explicit statements; the ML layer
exists because phrase lists miss paraphrase. Use both.

We deliberately do NOT enumerate, store, or surface specific methods of self-harm
anywhere. The screener detects *risk presence/severity*, not *method*, and method
text is redacted before anything is logged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Callable
import re
import hashlib
import datetime as dt


class RiskTier(IntEnum):
    NONE = 0          # no signal
    DISTRESS = 1      # elevated distress, no ideation -> gentle in-app support
    POSSIBLE = 2      # possible ideation -> show resources prominently, warm handoff
    EXPLICIT = 3      # explicit/imminent language -> resources + human review queue


# High-recall, intentionally broad. False positives here are acceptable.
# These detect PRESENCE of risk language, not methods. Keep method words OUT.
_EXPLICIT_PATTERNS = [
    r"\bkill myself\b", r"\bend my life\b", r"\bdon'?t want to (be alive|live)\b",
    r"\bbetter off (dead|without me)\b", r"\bwant to die\b", r"\bsuicid",
    r"\bno reason to (live|go on)\b", r"\b(take|end) my own life\b",
    r"\bcan'?t (do this|go on) (anymore|any longer)\b.*\b(alive|here)\b",
]
_POSSIBLE_PATTERNS = [
    r"\bwhat'?s the point\b", r"\bhopeless\b", r"\beveryone.*better off\b",
    r"\bcan'?t see a (future|way out)\b", r"\bdisappear\b", r"\btired of (living|everything)\b",
    r"\bgive up\b", r"\bnothing matters\b", r"\bburden to everyone\b",
]
_EXPLICIT_RE = re.compile("|".join(_EXPLICIT_PATTERNS), re.I)
_POSSIBLE_RE = re.compile("|".join(_POSSIBLE_PATTERNS), re.I)


@dataclass
class CrisisAssessment:
    tier: RiskTier
    phrase_hit: bool
    model_risk: float                 # [0,1] from an optional risk classifier
    reasons: list[str] = field(default_factory=list)
    # text is NOT stored here; see redact() and the audit record below.


def assess(text: str, model_risk: float = 0.0) -> CrisisAssessment:
    """
    Combine a high-recall phrase layer with an optional ML risk score.
    `model_risk` should come from a *separately trained, clinically reviewed*
    risk classifier if you have one; default 0.0 means phrase-layer only.
    """
    reasons: list[str] = []
    explicit = bool(_EXPLICIT_RE.search(text or ""))
    possible = bool(_POSSIBLE_RE.search(text or ""))

    if explicit:
        reasons.append("explicit_risk_language")
    if possible:
        reasons.append("possible_risk_language")
    if model_risk >= 0.85:
        reasons.append("model_high_risk")
    elif model_risk >= 0.5:
        reasons.append("model_elevated_risk")

    # Tier assignment — OR logic, biased toward sensitivity.
    if explicit or model_risk >= 0.85:
        tier = RiskTier.EXPLICIT
    elif possible or model_risk >= 0.5:
        tier = RiskTier.POSSIBLE
    else:
        tier = RiskTier.NONE  # DISTRESS tier is set by the caller from emotion score

    return CrisisAssessment(tier, explicit or possible, model_risk, reasons)


def redact(text: str) -> str:
    """Best-effort redaction before any logging. Drops anything that looks like a
    described method and keeps only a hash for dedup. Never log raw crisis text
    unless the user has explicitly consented to a safety record."""
    return hashlib.sha256((text or "").encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Response policy. Returns an action plan; the app layer renders it.
# ---------------------------------------------------------------------------
@dataclass
class ContactConsent:
    """Pre-authorized by the user, stored explicitly, revocable any time."""
    contact_label: str                  # e.g. "my sister Priya"
    channel: str                        # "sms" | "email" | "push-to-contact"
    allow_tier: RiskTier = RiskTier.EXPLICIT
    user_can_cancel_window_sec: int = 300   # user sees + can stop the alert first


@dataclass
class CrisisAction:
    show_resources: bool
    prominent: bool                     # take over the screen vs. gentle banner
    offer_warm_handoff: bool            # one-tap call/text/chat
    queue_for_human_review: bool        # staffed safety team, if you have one
    propose_contact_alert: Optional[ContactConsent]  # NOT sent yet — proposed to user
    message: str


def plan_response(assessment: CrisisAssessment,
                  consent: Optional[ContactConsent],
                  distress_tier_1: bool = False) -> CrisisAction:
    t = assessment.tier
    if t == RiskTier.NONE and distress_tier_1:
        t = RiskTier.DISTRESS

    if t == RiskTier.NONE:
        return CrisisAction(False, False, False, False, None, "")

    if t == RiskTier.DISTRESS:
        return CrisisAction(
            show_resources=True, prominent=False, offer_warm_handoff=False,
            queue_for_human_review=False, propose_contact_alert=None,
            message=("It sounds like today's been heavy. If it would help to talk "
                     "to someone, support is available any time."),
        )

    if t == RiskTier.POSSIBLE:
        return CrisisAction(
            show_resources=True, prominent=True, offer_warm_handoff=True,
            queue_for_human_review=False, propose_contact_alert=None,
            message=("I'm really glad you wrote this down. You don't have to carry "
                     "this alone — talking to a trained counsellor can help, and "
                     "they're available right now."),
        )

    # EXPLICIT
    # We *propose* a contact alert ONLY if the user pre-consented for this tier.
    # The app must show the user that an alert is about to go out and let them
    # cancel it within consent.user_can_cancel_window_sec. Never silent.
    propose = consent if (consent and t >= consent.allow_tier) else None
    return CrisisAction(
        show_resources=True, prominent=True, offer_warm_handoff=True,
        queue_for_human_review=True, propose_contact_alert=propose,
        message=("It sounds like you're in a lot of pain right now, and I don't "
                 "want you to go through this by yourself. A trained counsellor "
                 "can talk with you right now — please reach out. You matter."),
    )


# ---------------------------------------------------------------------------
# Minimal, privacy-preserving audit record for the safety queue.
# Store the LEAST you can. No raw text unless explicitly consented.
# ---------------------------------------------------------------------------
@dataclass
class SafetyAuditRecord:
    user_id: str
    ts: str
    tier: int
    reasons: list[str]
    text_fingerprint: str               # from redact(); not the text itself
    resources_shown: bool
    contact_alert_proposed: bool
    contact_alert_sent: bool = False    # set true only after user did not cancel


def make_audit(user_id: str, a: CrisisAssessment, action: CrisisAction,
               raw_text: str) -> SafetyAuditRecord:
    return SafetyAuditRecord(
        user_id=user_id,
        ts=dt.datetime.now(dt.timezone.utc).isoformat(),
        tier=int(a.tier),
        reasons=a.reasons,
        text_fingerprint=redact(raw_text),
        resources_shown=action.show_resources,
        contact_alert_proposed=action.propose_contact_alert is not None,
    )
