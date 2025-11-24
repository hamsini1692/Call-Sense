# agents/frustration_loop.py

from typing import Any, Dict, List
import re
import json

from graph.state import CallState
from graph.tools import Tools
from graph.a2a import send_message
from typing import Any, Dict, List, Optional
from graph.tools import Tools


FRUSTRATION_LABELS = ["low", "medium", "high"]


# ---------------- rule-based fallback ---------------- #

def _rule_based_frustration(utterance: str) -> str:
    u = utterance.lower()

    high_triggers = [
        "this is the third time",
        "this is the second time",
        "i am very frustrated",
        "unacceptable",
        "i want to cancel",
        "close my account",
        "worst experience",
    ]

    medium_triggers = [
        "not happy",
        "disappointed",
        "still not working",
        "nobody helped me",
        "already tried",
        "taking too long",
    ]

    if any(t in u for t in high_triggers):
        return "high"
    if any(t in u for t in medium_triggers):
        return "medium"
    return "low"


# ---------------- prompt builder ---------------- #

def _build_frustration_prompt(utterances: List[str]) -> str:
    utterance_block = "\n".join(
        f"{idx+1}. {utt}" for idx, utt in enumerate(utterances)
    )

    return f"""
You are analyzing a customer support call.

Below are the customer's utterances in order:

{utterance_block}

For each line, label the customer's frustration level as one of:
- low
- medium
- high

Return a JSON array of objects with keys:
- index: integer
- utterance: text
- level: one of ["low","medium","high"]
""".strip()


# ---------------- helper: compute overall frustration ---------------- #

def _overall_level(timeline: List[Dict[str, Any]]) -> str:
    counts = {"low": 0, "medium": 0, "high": 0}

    for item in timeline:
        lvl = item.get("level", "low").lower()
        if lvl in counts:
            counts[lvl] += 1

    if counts["high"] > 0:
        return "high"
    if counts["medium"] > 0:
        return "medium"
    return "low"


# ---------------- main loop agent ---------------- #

def frustration_loop_agent(
    call_state: CallState,
    tools: Optional[Tools] = None,
) -> CallState:
    """
    Loop agent that:
    - iterates over utterances
    - builds frustration timeline (LLM or fallback)
    - then sends A2A message to PainPoints agent
    """
    utterances = call_state.utterances

    # If no segmentation happened yet, fallback to entire transcript
    if not utterances:
        text = call_state.cleaned_transcript or call_state.raw_transcript or ""
        utterances = [text] if text else []
        if not utterances:
            call_state.frustration_timeline = []
            call_state.step_count += 1
            return call_state

    llm = tools.get_llm() if tools else None
    timeline: List[Dict[str, Any]] = []

    # ---------------- LLM path ---------------- #
    if llm is not None:
        try:
            prompt = _build_frustration_prompt(utterances)
            call_state.tool_calls += 1

            completion = llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Classify frustration level per utterance. Output ONLY JSON."
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )

            raw = completion.choices[0].message.content.strip()
            call_state.tool_successes += 1

            parsed = json.loads(raw)

            if isinstance(parsed, list):
                # sanitize
                timeline = []
                for item in parsed:
                    lvl = (item.get("level") or "").lower()
                    if lvl not in FRUSTRATION_LABELS:
                        lvl = _rule_based_frustration(item.get("utterance", ""))
                    timeline.append(
                        {
                            "index": int(item.get("index", 0)),
                            "utterance": item.get("utterance", ""),
                            "level": lvl,
                        }
                    )

        except Exception:
            timeline = []

    # ---------------- Fallback path ---------------- #
    if not timeline:
        timeline = [
            {
                "index": i + 1,
                "utterance": utt,
                "level": _rule_based_frustration(utt),
            }
            for i, utt in enumerate(utterances)
        ]

    # Save to state
    call_state.frustration_timeline = timeline
    call_state.step_count += 1

    # ---------------- A2A Protocol message ---------------- #
    overall = _overall_level(timeline)
    high_segments = [item for item in timeline if item["level"] == "high"]

    send_message(
        call_state,
        from_agent="frustration_loop",
        to_agent="pain_points",
        msg_type="frustration_summary",
        payload={
            "overall_level": overall,
            "high_segments": high_segments,
            "timeline_length": len(timeline),
        },
    )

    return call_state
