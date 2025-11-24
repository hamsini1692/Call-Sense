# agents/sentiment.py

from typing import Any
import re
import json

from graph.state import CallState
from graph.tools import Tools
from graph.a2a import send_message
from typing import Any, Optional
from graph.tools import Tools


ALLOWED_LABELS = {
    "very_negative",
    "negative",
    "neutral",
    "positive",
    "very_positive",
    "mixed",
    "unknown",
}


# ---------- simple rule-based fallback ---------- #

NEGATIVE_HINTS = [
    "angry", "frustrated", "upset", "mad", "cancel", "close my account",
    "this is the third time", "this is the second time", "unacceptable",
    "disappointed", "complaint", "not happy", "terrible", "awful", "worst",
]

POSITIVE_HINTS = [
    "thank you", "thanks a lot", "appreciate", "great", "awesome",
    "helpful", "resolved", "perfect", "excellent",
]


def rule_based_sentiment(text: str) -> str:
    """
    Extremely lightweight heuristic sentiment classifier.
    This is only a safety net if the LLM/tool is unavailable.
    """
    if not text:
        return "unknown"

    t = text.lower()

    neg_hits = sum(1 for w in NEGATIVE_HINTS if w in t)
    pos_hits = sum(1 for w in POSITIVE_HINTS if w in t)

    if neg_hits > pos_hits and neg_hits >= 2:
        return "very_negative"
    if neg_hits > pos_hits and neg_hits >= 1:
        return "negative"
    if pos_hits > neg_hits and pos_hits >= 2:
        return "very_positive"
    if pos_hits > neg_hits and pos_hits >= 1:
        return "positive"
    if pos_hits > 0 and neg_hits > 0:
        return "mixed"

    return "neutral"


# ---------- prompt builder for LLM ---------- #

def _build_sentiment_prompt(call_state: CallState) -> str:
    transcript = call_state.cleaned_transcript or call_state.raw_transcript or ""
    summary = call_state.summary or ""

    return f"""
You classify the overall customer sentiment for a support call.

Transcript:
\"\"\"{transcript}\"\"\"

Summary:
\"\"\"{summary}\"\"\"

Valid labels:
- very_negative
- negative
- neutral
- positive
- very_positive
- mixed

Return ONLY the label.
""".strip()


def _normalize_label(label: str) -> str:
    """
    Normalize the model output into an allowed label.
    """
    if not label:
        return "unknown"

    first = re.split(r"\s+", label.strip())[0].lower()
    first = re.sub(r"[^a-z_]", "", first)

    return first if first in ALLOWED_LABELS else "unknown"


# ---------- main sentiment agent ---------- #

def sentiment_agent(
    call_state: CallState,
    tools: Optional[Tools] = None,
) -> CallState:
    """
    Sentiment & Emotion Detection Agent.

    OUTPUT:
        - call_state.sentiment (string)
        - A2A message sent to actions agent
    """
    text = call_state.cleaned_transcript or call_state.raw_transcript or ""
    if not text.strip():
        call_state.sentiment = "unknown"
        call_state.step_count += 1
        return call_state

    llm = tools.get_llm() if tools else None
    sentiment_label: str | None = None

    # ---------- LLM Path ----------
    if llm is not None:
        try:
            prompt = _build_sentiment_prompt(call_state)
            call_state.tool_calls += 1

            completion = llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise sentiment classifier. Return only the label."
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )

            raw_label = completion.choices[0].message.content.strip()
            call_state.tool_successes += 1

            sentiment_label = _normalize_label(raw_label)

        except Exception:
            sentiment_label = None

    # ---------- Fallback ----------
    if not sentiment_label:
        sentiment_label = rule_based_sentiment(text)

    call_state.sentiment = sentiment_label
    call_state.step_count += 1

    # ---------- A2A Protocol ----------
    # downstream: actions_agent may want to know sentiment
    send_message(
        call_state,
        from_agent="sentiment",
        to_agent="actions",
        msg_type="sentiment_signal",
        payload={"sentiment": sentiment_label},
    )

    return call_state
