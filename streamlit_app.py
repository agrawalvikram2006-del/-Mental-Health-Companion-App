"""
streamlit_app.py — Streamlit entry point for Mindful.

This is the file to set as the "Main file path" when deploying on
Streamlit Community Cloud (e.g. `mindful/streamlit_app.py`, or just
`streamlit_app.py` if the contents of the mindful/ folder are your repo root).

It reuses the same core logic as the FastAPI service (ml/, core/, config/) and
preserves the safety design:
  * crisis screening runs before analytics,
  * resources are shown generously to the user,
  * a trusted-contact alert is opt-in, shown first, and never auto-sent.

By default it runs on the lightweight lexicon fallback in ml/emotion.py so it
deploys on a free tier without downloading a multi-GB model. To use the real
fine-tuned BERT model, add `transformers` and `torch` to requirements.txt — note
the free Streamlit tier may not have the memory to load it.

NOTE: session history here lives in st.session_state and resets when the tab
closes. It is a demo store, not the encrypted per-user database a real product
needs.
"""

import os
import sys

# Make the sibling packages (ml/, core/, config/) importable regardless of where
# Streamlit Cloud runs this from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st

from ml.emotion import EmotionModel
from ml.trends import TrendTracker
from core.scoring import compose_wellbeing
from core.crisis import assess, plan_response, ContactConsent, RiskTier
from config.resources import for_country

st.set_page_config(page_title="Mindful — daily check-in", page_icon="🌤️", layout="centered")


@st.cache_resource
def get_model():
    return EmotionModel()


# ---- session state ----------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []          # list of wellbeing floats
if "consent_on" not in st.session_state:
    st.session_state.consent_on = False
if "pending_alert" not in st.session_state:
    st.session_state.pending_alert = None  # holds a proposed (not sent) alert

model = get_model()

# ---- header -----------------------------------------------------------------
st.title("Your inner weather")
st.caption("A quiet daily check-in. The shaded band is what steady looks like for you — "
           "not a target, just your own normal.")
st.divider()

# ---- check-in form ----------------------------------------------------------
with st.form("checkin", clear_on_submit=False):
    mood = st.slider("How are you today?", 1, 5, 3,
                     help="1 = really low · 5 = really good")
    note = st.text_area("Anything you'd like to note? (optional)", height=110,
                        placeholder="A sentence or two about your day…")
    country = st.selectbox("Region (for support resources)", ["IN", "US", "Other"], index=0)
    submitted = st.form_submit_button("Save check-in", use_container_width=True)

cc = {"IN": "IN", "US": "US", "Other": None}[country]

# ---- consent setting --------------------------------------------------------
with st.expander("Let one person know — only if you choose (off by default)"):
    st.write("If you turn this on, and a check-in ever suggests you're really struggling, "
             "we'll *offer* to message one person you pick. You always see it first and can "
             "stop it. Nothing is ever sent silently.")
    st.session_state.consent_on = st.toggle("Enable trusted-contact alerts",
                                            value=st.session_state.consent_on)
    if st.session_state.consent_on:
        st.text_input("Who should we offer to notify?", placeholder="e.g. my sister Priya",
                      key="contact_label")

# ---- process a submission ---------------------------------------------------
if submitted:
    # 1) emotion + 2) crisis screen FIRST
    emo = model.predict(note or "")
    crisis = assess(note or "", model_risk=0.0)
    consent = None
    if st.session_state.consent_on:
        consent = ContactConsent(st.session_state.get("contact_label") or "a trusted contact", "sms")
    action = plan_response(crisis, consent, distress_tier_1=(emo.distress >= 0.6))

    # 3) wellbeing + 4) trend
    wellbeing = compose_wellbeing(mood, emo.distress)
    tracker = TrendTracker(history=st.session_state.history)
    signal = tracker.observe(wellbeing)
    st.session_state.history = tracker.history  # persist

    # stash a proposed (not sent) alert for the confirm step
    if action.propose_contact_alert is not None:
        st.session_state.pending_alert = action.propose_contact_alert.contact_label
    else:
        st.session_state.pending_alert = None

    # ---- support comes first ----
    if action.show_resources:
        box = st.error if action.prominent else st.info
        box(action.message)
        cols = st.columns(len(for_country(cc)))
        for col, r in zip(cols, for_country(cc)):
            with col:
                st.markdown(f"**{r.name}**")
                if r.call:
                    st.markdown(f"📞 **{r.call}**")
                if r.text:
                    st.markdown(f"💬 text **{r.text}**")
                if r.chat_url:
                    st.markdown(f"[Open chat]({r.chat_url})")
                st.caption(r.note)

    # ---- the consent-gated alert: proposed, never auto-sent ----
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

    # ---- gentle reflection ----
    if not action.prominent:
        st.success("Saved. Thanks for checking in.")
    sev = signal.severity
    if sev in ("notable", "marked"):
        st.write("Your recent check-ins look tougher than your usual. That's worth being "
                 "kind to yourself about — and a counsellor or someone you trust can help carry it.")

# ---- trend chart (baseline band + line) -------------------------------------
st.divider()
hist = st.session_state.history
if len(hist) >= 2:
    df = pd.DataFrame({"day": list(range(1, len(hist) + 1)), "wellbeing": hist})
    if len(hist) >= 7:
        import statistics
        prior = hist[:-1]
        med = statistics.median(prior)
        mad = max(0.05, statistics.median([abs(x - med) for x in prior]))
        spread = mad * 1.4826 * 1.5
        df["lo"] = max(0.0, med - spread)
        df["hi"] = min(1.0, med + spread)
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
st.caption("Mindful is a wellbeing companion — not a medical device, a diagnosis, or a "
           "substitute for professional care. In an emergency, contact local emergency services.")
