# Mindful — a daily check-in wellbeing companion (reference design)

A reference architecture and starter implementation for a daily mental-health
check-in app that learns each user's baseline mood, surfaces trends and likely
triggers, offers gentle suggestions, and handles crisis signals carefully.

**This is not a medical device, a diagnostic tool, or a substitute for
professional care.** Several decisions below exist specifically to keep it from
pretending otherwise. Read the "Safety design" section before you build the
crisis or trusted-contact features — those are where a naive implementation
hurts people.

---

## How a check-in flows

```
mood tap (1–5) + optional note
        │
        ▼
[1] emotion model  ──►  distress score (ml/emotion.py)
        │
        ▼
[2] crisis screen FIRST  ──►  tier 0–3 (core/crisis.py)      ← safety before analytics
        │
        ▼
[3] compose wellbeing index (core/scoring.py)
        │
        ▼
[4] update personal baseline + deviation (ml/trends.py)
        │
        ▼
[5] response: gentle suggestion  +  (if needed) resources / warm handoff
        │
        ▼
[6] minimal audit record if any risk tier fired
```

Crisis screening runs **before** the analytics so that a user in distress is met
with support, not a mood chart.

## Components

| File | Role |
|---|---|
| `ml/emotion.py` | Transformer emotion classifier → emotion buckets + distress score. Fine-tune scaffold included. Degrades to a lexicon fallback so you can develop without a GPU. |
| `ml/trends.py` | Robust per-user baseline (median + MAD), EWMA, robust z-score, CUSUM changepoint. **Optional** LSTM forecaster — forecast only, never the sole basis for an alert. |
| `core/scoring.py` | Explicit, auditable composition of the wellbeing index. Includes PHQ-9 / GAD-7 severity bands. |
| `core/crisis.py` | Layered crisis screening + tiered, consent-gated, human-reviewed response policy. |
| `config/resources.py` | Region-aware crisis resources (verified June 2026). Wire to a maintained source. |
| `api/app.py` | FastAPI surface wiring it together. |
| `dashboard.jsx` | Calm dashboard: baseline-band trend, triggers, suggestion, consent toggle, support view. |

## Run the core (no GPU / no network needed)

```bash
python3 -c "import sys;sys.path.insert(0,'.'); \
from ml.emotion import EmotionModel; from ml.trends import TrendTracker; \
from core.scoring import compose_wellbeing; from core.crisis import assess, plan_response; \
print(assess('I want to die').tier.name)"   # -> EXPLICIT
```

For the real classifier: `pip install transformers torch` and the model in
`emotion.py` downloads automatically. For the API: `pip install fastapi uvicorn`
then `uvicorn api.app:app --reload`.

---

## Safety design (the part that matters most)

### Two decisions, two error tolerances

- **"Show this user help?"** → favor sensitivity. Showing a helpline to someone
  who didn't strictly need it costs little. Over-trigger on purpose.
- **"Tell another human about this user?"** → favor specificity, require explicit
  consent and human review. Disclosing someone's mental state to a third party
  without informed, revocable consent is a privacy violation and can shatter the
  trust that keeps them using the app at all.

### What the crisis system does and does not do

- **Layered detection:** a high-recall phrase layer (catches explicit statements
  ML misses) combined with an optional, clinically-reviewed risk classifier
  (catches paraphrase the phrase list misses). Use both; neither alone.
- **Tiered response:** distress → gentle in-app support; possible ideation →
  prominent resources + one-tap warm handoff; explicit/imminent → resources +
  human-review queue, and a trusted-contact alert **only if pre-consented**,
  **shown to the user first**, and **cancellable** before it sends.
- **No automated emergency/police dispatch.** Ever. Auto-dispatch on a classifier
  output endangers the people it claims to protect and is a known harm of this
  product category. Coordinated response, when truly needed, is the job of trained
  crisis lines (988 / Tele-MANAS), which escalate conservatively.
- **No method detail, anywhere.** The screener detects risk *presence/severity*,
  not *method*. Raw crisis text is fingerprinted (hashed), not stored, unless the
  user has explicitly consented to a safety record. Don't build features that
  elicit or display methods.

### Models, honestly

- A text emotion classifier detects *expressed emotion*, which correlates with
  but is not the same as clinical depression or anxiety **severity**. Don't label
  a user "depressed" from text. For severity, administer validated instruments
  (**PHQ-9**, **GAD-7**) on a cadence and treat NLP as one corroborating signal.
  PHQ-9 item 9 (self-harm) routes into the crisis flow regardless of total score.
- For trend detection on sparse daily check-ins, robust statistics (EWMA + robust
  z-score + CUSUM) are more reliable and far more interpretable than an LSTM. The
  LSTM is included as an optional forecaster for long, dense histories — it never
  gates an alert by itself.

## Privacy, consent & compliance (not legal advice)

Mental-health data is **special category** data under GDPR and sensitive under
most regimes. Before launch, with an actual lawyer and clinician:

- Encrypt at rest with per-user keys; minimize what you store; set short retention.
- Make consent granular, explicit, and revocable — especially for any third-party
  alert. Default it **off**.
- If you operate as or with a US covered entity, **HIPAA** may apply. If you serve
  minors (you list college students), check **COPPA**/age rules and local consent.
- Be clear in-product that this is a wellbeing tool, not treatment, and show local
  emergency guidance.
- A staffed safety queue is a real operational commitment. If you offer human
  review, you must actually staff it; a queue no one reads is worse than none.

## Crisis resources (verified June 2026 — keep these fresh)

- **India — Tele-MANAS:** call **14416** (or 1800-891-4416). Free, 24/7, 20+ languages.
- **US — 988 Suicide & Crisis Lifeline:** call or text **988**; chat at 988lifeline.org.
- **Anywhere — Find A Helpline:** https://findahelpline.com — verified lines for
  130+ countries; recommended as your canonical, maintained source so a user is
  never shown a dead number.

Helpline details change. Wire `config/resources.py` to a maintained source on a
schedule rather than trusting a hardcoded list forever.

---

*If you're reading this and going through a hard time yourself: the numbers above
are for you too.*
