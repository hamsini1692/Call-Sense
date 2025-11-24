# agents/pain_points.py

from typing import List, Dict, Any
import json

from graph.state import CallState
from graph.tools import Tools
from graph.a2a import get_messages_for_agent
from typing import List, Dict, Any, Optional
from graph.tools import Tools


def _rule_based_pain_points(call_state: CallState) -> List[str]:
    """
    Heuristic-based pain point extractor (fallback).
    Uses entities, transcript text, and A2A frustration summary.
    """
    entities = call_state.entities or {}
    text = (call_state.cleaned_transcript or call_state.raw_transcript or "").lower()
    pain_points: List[str] = []

    issue = entities.get("issue")
    product = entities.get("product")

    if issue:
        if product:
            pain_points.append(f"{issue} related to {product}")
        else:
            pain_points.append(issue)

    # Use text patterns
    if "refund" in text or "chargeback" in text:
        pain_points.append("refund or chargeback delay")
    if "login" in text or "password" in text:
        pain_points.append("login or authentication issues")
    if "fee" in text or "overcharged" in text:
        pain_points.append("unexpected fees or overcharging")

    # 🔹 Use A2A frustration summary
    msgs = get_messages_for_agent(
        call_state,
        agent_name="pain_points",
        msg_type="frustration_summary",
    )
    if msgs:
        summary = msgs[-1].get("payload", {})  # take latest
        if summary.get("overall_level") == "high":
            pain_points.append("customer is highly frustrated after multiple attempts")

    # dedupe while preserving order
    seen = set()
    deduped: List[str] = []
    for p in pain_points:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    if not deduped:
        deduped = ["unclear primary pain point"]

    return deduped


def _build_pain_point_prompt(
    call_state: CallState,
    frustration_summary: Dict[str, Any],
) -> str:
    transcript = call_state.cleaned_transcript or call_state.raw_transcript or ""
    summary = call_state.summary or ""
    entities = call_state.entities or {}
    frustration_summary_json = json.dumps(frustration_summary, indent=2)

    return f"""
You are an assistant that extracts customer pain points from support calls.

Transcript:
\"\"\"{transcript}\"\"\"

Internal summary:
\"\"\"{summary}\"\"\"

Entities/context:
{json.dumps(entities, indent=2)}

Frustration summary from another agent:
{frustration_summary_json}

Task:
- Identify 2–5 distinct customer pain points.
- Each pain point should be a short phrase (5–12 words).
- Do not repeat the same idea with different wording.

Return a JSON array of strings.
""".strip()


def pain_points_agent(
    call_state: CallState,
    tools: Optional[Tools] = None,
) -> CallState:
    """
    Pain Point & Root Cause Agent.

    INPUT:
        - summary, entities, frustration_timeline
        - A2A message from frustration_loop agent

    OUTPUT:
        - call_state.pain_points: List[str]
    """
    llm = tools.get_llm() if tools is not None else None
    pain_points: List[str] | None = None

    # get A2A frustration summary (if any)
    msgs = get_messages_for_agent(
        call_state,
        agent_name="pain_points",
        msg_type="frustration_summary",
    )
    frustration_summary: Dict[str, Any] = msgs[-1].get("payload", {}) if msgs else {}

    if llm is not None:
        try:
            prompt = _build_pain_point_prompt(call_state, frustration_summary)
            call_state.tool_calls += 1

            completion = llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You extract concise, non-overlapping customer pain points.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            content = completion.choices[0].message.content.strip()
            call_state.tool_successes += 1

            parsed = json.loads(content)
            if isinstance(parsed, list):
                pain_points = [str(p).strip() for p in parsed if str(p).strip()]
        except Exception:
            pain_points = None

    # Fallback to rule-based if LLM is unavailable or fails
    if not pain_points:
        pain_points = _rule_based_pain_points(call_state)

    call_state.pain_points = pain_points
    call_state.step_count += 1

    return call_state
