"""
streamlit_app.py — Mindful daily check-in (SINGLE-FILE version).

Everything is inlined here on purpose: there are NO imports from local folders,
so deploying on Streamlit Community Cloud only needs this one file plus a
requirements.txt containing `streamlit`. Set this file as the Main file path.

Safety design is preserved:
  * crisis screening runs before analytics,
  * support resources are shown generously to the user,
  * a trusted-contact alert is opt-in, shown first, and never auto-sent,
  * no automated emergency dispatch.

This runs on a lightweight lexicon emotion model so it deploys anywhere. To use
a real fine-tuned BERT model you would reintroduce transformers/torch and a
proper backend — see the project report. Session history lives in memory and
resets when the tab closes; it is a demo store, not a real database.

Mindful is a wellbeing companion, not a medical device, diagnosis, or a
substitute for professional care.
"""

import re
import math
import statistics
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Mindful — daily check-in", page_icon="🌤️", layout="centered")

# ============================================================================
#  EMOTION (lightweight lexicon model)
# ============================================================================
CORE_EMOTIONS = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
_DISTRESS_WEIGHTS = {"sadness": 1.0, "fear": 0.9, "anger": 0.6, "disgust": 0.5}
_LEX = {
    "sadness": ["sad", "down", "empty", "hopeless", "worthless", "tired", "lonely",
                "cry", "exhausted", "numb", "low", "miserable"],
    "fear": ["anxious", "worried", "scared", "panic", "nervous", "afraid", "stressed",
             "overwhelmed", "dread", "tense"],
    "anger": ["angry", "furious", "hate", "frustrated", "annoyed", "irritated", "resent"],
    "joy": ["happy", "good", "great", "grateful", "excited", "calm", "fine", "okay",
            "content", "relaxed", "hopeful"],
}


@dataclass
class EmotionResult:
    emotions: dict
    distress: float
    dominant: str


def predict_emotion(text: str) -> EmotionResult:
    t = (text or "").lower()
    buckets = {e: 0.0 for e in CORE_EMOTIONS}
    for emo, words in _LEX.items():
        buckets[emo] = sum(t.count(w) for w in words)
    if sum(buckets.values()) == 0:
        buckets["neutral"] = 1.0
    total = sum(buckets.values()) or 1.0
    buckets = {k: v / total for k, v in buckets.items()}
    distress = sum(buckets[e] * w for e, w in _DISTRESS_WEIGHTS.items())
    distress = max(0.0, min(1.0, distress))
    return EmotionResult(buckets, distress, max(buckets, key=buckets.get))


# ============================================================================
#  SCORING
# ============================================================================
def compose_wellbeing(mood: int, text_distress: Optional[float] = None) -> float:
    m = (max(1, min(5, mood)) - 1) / 4.0
    parts = [(m, 0.7)]
    if text_distress is not None:
        parts.append((1.0 - max(0.0, min(1.0, text_distress)), 0.3))
    wsum = sum(w for _, w in parts)
    return sum(v * w for v, w in parts) / wsum


# ============================================================================
#  TREND (robust baseline + EWMA + z-score + CUSUM)
# ============================================================================
_MAD_TO_STD = 1.4826


@dataclass
class DeviationSignal:
    value: float
    robust_z: Optional[float]
    ewma: float
    sustained_drop: bool
    severity: str


class TrendTracker:
    def __init__(self, history=None, alpha=0.3, min_baseline=7, k=0.5, h=4.0):
        self.history = list(history or [])
        self.alpha, self.min_baseline, self.k, self.h = alpha, min_baseline, k, h
        self._cusum, self._ewma = 0.0, None
        for v in self.history:
            self._ewma = v if self._ewma is None else alpha * v + (1 - alpha) * self._ewma

    def observe(self, w: float) -> DeviationSignal:
        w = max(0.0, min(1.0, w))
        prior = self.history[:]
        base = None
        if len(prior) >= self.min_baseline:
            med = statistics.median(prior)
            mad = max(0.05, statistics.median([abs(x - med) for x in prior]))
            base = (med, mad)
        self.history.append(w)
        self._ewma = w if self._ewma is None else self.alpha * w + (1 - self.alpha) * self._ewma
        z = None
        if base:
            med, mad = base
            z = (w - med) / (mad * _MAD_TO_STD)
            self._cusum = min(0.0, self._cusum + z + self.k)
        sustained = self._cusum < -self.h
        if sustained:
            self._cusum = 0.0
        return DeviationSignal(w, z, self._ewma, sustained, _severity(z, sustained))


def _severity(z, sustained):
    if z is None:
        return "none"
    if z <= -3.0 or sustained:
        return "marked"
    if z <= -2.0:
        return "notable"
    if z <= -1.5:
        return "mild"
    return "none"


# ============================================================================
#  CRISIS (layered, conservative, consent-gated)
# ============================================================================
class RiskTier(IntEnum):
    NONE = 0
    DISTRESS = 1
    POSSIBLE = 2
    EXPLICIT = 3


_EXPLICIT = re.compile("|".join([
    r"\bkill myself\b", r"\bend my life\b", r"\bdon'?t want to (be alive|live)\b",
    r"\bbetter off (dead|without me)\b", r"\bwant to die\b", r"\bsuicid",
    r"\bno reason to (live|go on)\b", r"\b(take|end) my own life\b",
]), re.I)
_POSSIBLE = re.compile("|".join([
    r"\bwhat'?s the point\b", r"\bhopeless\b", r"\beveryone.*better off\b",
    r"\bcan'?t see a (future|way out)\b", r"\bdisappear\b", r"\btired of (living|everything)\b",
    r"\bgive up\b", r"\bnothing matters\b", r"\bburden to everyone\b",
]), re.I)


@dataclass
class ContactConsent:
    contact_label: str
    allow_tier: RiskTier = RiskTier.EXPLICIT


def assess_crisis(text: str) -> RiskTier:
    text = text or ""
    if _EXPLICIT.search(text):
        return RiskTier.EXPLICIT
    if _POSSIBLE.search(text):
        return RiskTier.POSSIBLE
    return RiskTier.NONE


@dataclass
class CrisisAction:
    show_resources: bool
    prominent: bool
    message: str
    propose_contact: Optional[str]


def plan_response(tier: RiskTier, consent: Optional[ContactConsent], distress_t1: bool) -> CrisisAction:
    if tier == RiskTier.NONE and distress_t1:
        tier = RiskTier.DISTRESS
    if tier == RiskTier.NONE:
        return CrisisAction(False, False, "", None)
    if tier == RiskTier.DISTRESS:
        return CrisisAction(True, False,
            "It sounds like today's been heavy. If it would help to talk to someone, "
            "support is available any time.", None)
    if tier == RiskTier.POSSIBLE:
        return CrisisAction(True, True,
            "I'm really glad you wrote this down. You don't have to carry this alone — "
            "talking to a trained counsellor can help, and they're available right now.", None)
    propose = consent.contact_label if (consent and tier >= consent.allow_tier) else None
    return CrisisAction(True, True,
        "It sounds like you're in a lot of pain right now, and I don't want you to go through "
        "this by yourself. A trained counsellor can talk with you right now — please reach out. "
        "You matter.", propose)


# ============================================================================
#  RESOURCES (verified June 2026; keep wired to a maintained source in production)
# ============================================================================
RESOURCES = {
    "IN": [{"name": "Tele-MANAS (India)", "call": "14416", "text": None,
            "chat": None, "note": "Free · 24/7 · 20+ languages"}],
    "US": [{"name": "988 Lifeline (US)", "call": "988", "text": "988",
            "chat": "https://988lifeline.org/chat/", "note": "Free · confidential · 24/7"}],
}
GLOBAL = [{"name": "Find A Helpline", "call": None, "text": None,
           "chat": "https://findahelpline.com", "note": "Verified lines, 130+ countries"}]


def for_country(cc):
    return RESOURCES.get((cc or "").upper(), []) + GLOBAL


# ============================================================================
#  SESSION STATE
# ============================================================================
st.session_state.setdefault("history", [])
st.session_state.setdefault("consent_on", False)
st.session_state.setdefault("pending_alert", None)

# ============================================================================
#  UI
# ============================================================================
st.title("Your inner weather")
st.caption("A quiet daily check-in. The shaded band is what steady looks like for you — "
           "not a target, just your own normal.")
st.info("Prototype for demonstration — not a real support service or a substitute for care.",
        icon="ℹ️")
st.divider()

with st.form("checkin"):
    mood = st.slider("How are you today?", 1, 5, 3, help="1 = really low · 5 = really good")
    note = st.text_area("Anything you'd like to note? (optional)", height=110,
                        placeholder="A sentence or two about your day…")
    country = st.selectbox("Region (for support resources)", ["IN", "US", "Other"], index=0)
    submitted = st.form_submit_button("Save check-in", use_container_width=True)

cc = {"IN": "IN", "US": "US", "Other": None}[country]

with st.expander("Let one person know — only if you choose (off by default)"):
    st.write("If you turn this on, and a check-in ever suggests you're really struggling, we'll "
             "*offer* to message one person you pick. You always see it first and can stop it. "
             "Nothing is ever sent silently.")
    st.session_state.consent_on = st.toggle("Enable trusted-contact alerts",
                                            value=st.session_state.consent_on)
    if st.session_state.consent_on:
        st.text_input("Who should we offer to notify?", placeholder="e.g. my sister Priya",
                      key="contact_label")

if submitted:
    emo = predict_emotion(note or "")
    tier = assess_crisis(note or "")
    consent = ContactConsent(st.session_state.get("contact_label") or "a trusted contact") \
        if st.session_state.consent_on else None
    action = plan_response(tier, consent, distress_t1=(emo.distress >= 0.6))

    wellbeing = compose_wellbeing(mood, emo.distress)
    tracker = TrendTracker(history=st.session_state.history)
    signal = tracker.observe(wellbeing)
    st.session_state.history = tracker.history
    st.session_state.pending_alert = action.propose_contact

    if action.show_resources:
        (st.error if action.prominent else st.info)(action.message)
        res = for_country(cc)
        for col, r in zip(st.columns(len(res)), res):
            with col:
                st.markdown(f"**{r['name']}**")
                if r["call"]:
                    st.markdown(f"📞 **{r['call']}**")
                if r["text"]:
                    st.markdown(f"💬 text **{r['text']}**")
                if r["chat"]:
                    st.markdown(f"[Open chat]({r['chat']})")
                st.caption(r["note"])

    if st.session_state.pending_alert:
        st.warning(f"Because this check-in looked hard, we can let "
                   f"**{st.session_state.pending_alert}** know you might appreciate a message. "
                   f"You decide.")
        c1, c2 = st.columns(2)
        if c1.button("Send the message", use_container_width=True):
            st.success("Sent. (Demo — no real message dispatched.)")
            st.session_state.pending_alert = None
        if c2.button("Don't send", use_container_width=True):
            st.info("Nothing was sent.")
            st.session_state.pending_alert = None

    if not action.prominent:
        st.success("Saved. Thanks for checking in.")
    if signal.severity in ("notable", "marked"):
        st.write("Your recent check-ins look tougher than your usual. That's worth being kind to "
                 "yourself about — and a counsellor or someone you trust can help carry it.")

# ---- trend chart ------------------------------------------------------------
st.divider()
hist = st.session_state.history
if len(hist) >= 2:
    df = pd.DataFrame({"day": list(range(1, len(hist) + 1)), "wellbeing": hist})
    if len(hist) >= 7:
        prior = hist[:-1]
        med = statistics.median(prior)
        mad = max(0.05, statistics.median([abs(x - med) for x in prior]))
        spread = mad * _MAD_TO_STD * 1.5
        df["lo"], df["hi"] = max(0.0, med - spread), min(1.0, med + spread)
        try:
            import altair as alt
            band = alt.Chart(df).mark_area(opacity=0.25, color="#9DB5A6").encode(
                x="day:Q", y="lo:Q", y2="hi:Q")
            line = alt.Chart(df).mark_line(color="#2F4858", strokeWidth=2.5).encode(
                x=alt.X("day:Q", title="check-in"),
                y=alt.Y("wellbeing:Q", title="wellbeing", scale=alt.Scale(domain=[0, 1])))
            st.altair_chart((band + line).properties(height=260), use_container_width=True)
            st.caption("The line leaving the band is what a deviation from your normal looks like.")
        except Exception:
            st.line_chart(df.set_index("day")[["wellbeing"]])
    else:
        st.line_chart(df.set_index("day")[["wellbeing"]])
        st.caption("A personal baseline appears once you have about a week of check-ins.")
else:
    st.caption("Your trend will appear here after a couple of check-ins.")

st.divider()
st.caption("Mindful is a wellbeing companion — not a medical device, a diagnosis, or a substitute "
           "for professional care. In an emergency, contact local emergency services.")
