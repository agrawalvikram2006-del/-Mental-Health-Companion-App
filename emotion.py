"""
emotion.py — text -> emotion / distress signals.

WHAT THIS IS:  a wrapper around a transformer classifier that maps a check-in
text to (a) emotion probabilities and (b) a coarse distress score in [0, 1].

WHAT THIS IS NOT:  a diagnosis of depression or anxiety. Expressed emotion in
text is correlated with, but is not the same as, clinical severity. For severity,
administer validated self-report scales (PHQ-9, GAD-7) on a cadence and treat the
NLP signal as *one* corroborating input. See core/scales.py.

MODEL CHOICE:
  - Default inference uses a publicly available emotion model
    (j-hartmann/emotion-english-distilroberta-base or a GoEmotions BERT).
  - For your own data, fine-tune with `train()` below on labelled check-ins.
    Label with care and clinical input; do NOT scrape labels.

This file is written to run once `pip install transformers torch` and a model
are available. It is offline-safe in that it degrades to a lexicon fallback if
no model is present, so the rest of the system can be developed without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math

# Emotions we surface. Maps many fine-grained labels down to these buckets.
CORE_EMOTIONS = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]

# Emotions that contribute to the coarse "distress" score, with weights.
_DISTRESS_WEIGHTS = {"sadness": 1.0, "fear": 0.9, "anger": 0.6, "disgust": 0.5}

# Mapping from common model label sets -> our buckets.
_LABEL_MAP = {
    "joy": "joy", "happiness": "joy", "love": "joy", "optimism": "joy",
    "sadness": "sadness", "grief": "sadness", "disappointment": "sadness",
    "remorse": "sadness", "anger": "anger", "annoyance": "anger", "rage": "anger",
    "fear": "fear", "nervousness": "fear", "anxiety": "fear",
    "disgust": "disgust", "surprise": "surprise", "neutral": "neutral",
}


@dataclass
class EmotionResult:
    emotions: dict[str, float]          # bucketed probabilities, sums ~1.0
    distress: float                     # [0,1], higher = more distress
    dominant: str
    used_fallback: bool = False
    raw: Optional[dict] = field(default=None, repr=False)


class EmotionModel:
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        self.model_name = model_name
        self._pipe = None
        try:
            from transformers import pipeline  # type: ignore
            self._pipe = pipeline(
                "text-classification", model=model_name,
                top_k=None, truncation=True,
            )
        except Exception:
            # No model available — fall back to a small lexicon so dev can proceed.
            self._pipe = None

    def predict(self, text: str) -> EmotionResult:
        text = (text or "").strip()
        if not text:
            return EmotionResult({e: 0.0 for e in CORE_EMOTIONS}, 0.0, "neutral", True)
        if self._pipe is None:
            return self._lexicon_fallback(text)

        scores = self._pipe(text)[0]  # list of {label, score}
        buckets = {e: 0.0 for e in CORE_EMOTIONS}
        for item in scores:
            bucket = _LABEL_MAP.get(item["label"].lower(), "neutral")
            buckets[bucket] += float(item["score"])
        total = sum(buckets.values()) or 1.0
        buckets = {k: v / total for k, v in buckets.items()}
        distress = sum(buckets[e] * w for e, w in _DISTRESS_WEIGHTS.items())
        dominant = max(buckets, key=buckets.get)
        return EmotionResult(buckets, _clip01(distress), dominant, False, {"scores": scores})

    # --- dev-only fallback; never ship this as the real classifier -----------
    def _lexicon_fallback(self, text: str) -> EmotionResult:
        t = text.lower()
        lex = {
            "sadness": ["sad", "down", "empty", "hopeless", "worthless", "tired", "lonely", "cry"],
            "fear": ["anxious", "worried", "scared", "panic", "nervous", "afraid", "stressed"],
            "anger": ["angry", "furious", "hate", "frustrated", "annoyed"],
            "joy": ["happy", "good", "great", "grateful", "excited", "calm", "fine"],
        }
        buckets = {e: 0.0 for e in CORE_EMOTIONS}
        for emo, words in lex.items():
            buckets[emo] = sum(t.count(w) for w in words)
        if sum(buckets.values()) == 0:
            buckets["neutral"] = 1.0
        total = sum(buckets.values()) or 1.0
        buckets = {k: v / total for k, v in buckets.items()}
        distress = sum(buckets[e] * w for e, w in _DISTRESS_WEIGHTS.items())
        return EmotionResult(buckets, _clip01(distress), max(buckets, key=buckets.get), True)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ---------------------------------------------------------------------------
# Fine-tuning scaffold. Run on a machine with GPU + your labelled data.
# Expects a HuggingFace `datasets` Dataset with columns: "text", "label" (int).
# ---------------------------------------------------------------------------
def train(train_ds, eval_ds, label_names: list[str],
          base_model: str = "bert-base-uncased", out_dir: str = "./emotion-ft"):
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer, DataCollatorWithPadding)
    import numpy as np
    import evaluate

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(label_names),
        id2label=dict(enumerate(label_names)),
        label2id={l: i for i, l in enumerate(label_names)},
    )

    def _tok(batch):
        return tok(batch["text"], truncation=True, max_length=256)

    train_ds = train_ds.map(_tok, batched=True)
    eval_ds = eval_ds.map(_tok, batched=True)
    f1 = evaluate.load("f1")

    def _metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return f1.compute(predictions=preds, references=p.label_ids, average="macro")

    args = TrainingArguments(
        output_dir=out_dir, learning_rate=2e-5, num_train_epochs=4,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1",
        weight_decay=0.01,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        tokenizer=tok, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=_metrics,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir
