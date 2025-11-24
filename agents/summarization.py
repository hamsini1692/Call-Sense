# agents/summarization.py

from typing import Any, Dict
from textwrap import shorten
import json

from graph.state import CallState
from graph.tools import Tools
from graph.a2a import get_messages_for_agent
from typing import Any, Dict, Optional
from graph.tools import Tools



def _build_summary_prompt(
    call_state: CallState,
    entity_summary: Dict[str, Any],
) -> str:
    """
    Build a rich prompt for the LLM using:
    - cleaned transcript
    - extracted entities
    - compact A2A entity summary (from entities agent)
    """
    transcript = call_state.cleaned_transcript or call_state.raw_transcript or ""
    entities = call_state.entities or {}

    return f"""
You are an assistant that summarizes customer support calls for an operations team.

Transcript:
\"\"\"{transcript}\"\"\"

Extracted entities and context (if any):
{json.dumps(entities, indent=2)}

Entity summary from another agent:
{json.dumps(entity_summary, indent=2)}

Write a concise, NEUTRAL summary of this call for an internal CRM note.

Requirements:
- 4–6 sentences
- Mention the customer’s main issue and key context (prior attempts, deadlines, escalation, etc.)
- Mention the product or service if clear
- Capture the outcome (resolved vs unresolved) if it can be inferred
- Avoid copying long phrases verbatim; paraphrase instead.

Return plain text, no bullet points, no markdown.
""".strip()


def _rule_based_summary(call_state: CallState, max_chars: int = 500) -> str:
    """
    Very simple fallback summarizer if the LLM/tool is unavailable.
    """
    text = (
        call_state.cleaned_transcript
        or call_state.raw_transcript
        or ""
    )

    if not text.strip():
        return "No transcript content was available to summarize."

    # Naive heuristic: just truncate the beginning of the call with ellipsis.
    return shorten(text, width=max_chars, placeholder="...")


def summarization_agent(
    call_state: CallState,
    tools: Optional[Tools] = None,
) -> CallState:
    """
    Summarization Agent.

    INPUT:
        - call_state.cleaned_transcript (preferred)
        - call_state.raw_transcript (fallback)
        - call_state.entities (to enrich the summary)
        - A2A entity_summary message from entities_agent (if present)

    OUTPUT:
        - call_state.summary: a concise textual summary of the call
    """
    # If we literally have no text, bail early
    if not (call_state.cleaned_transcript or call_state.raw_transcript):
        call_state.summary = "No transcript content was available to summarize."
        call_state.step_count += 1
        return call_state

    llm = tools.get_llm() if tools is not None else None
    summary_text: str | None = None

    # Get A2A entity summary, if any
    msgs = get_messages_for_agent(
        call_state,
        agent_name="summarization",
        msg_type="entity_summary",
    )
    entity_summary: Dict[str, Any] = msgs[-1].get("payload", {}) if msgs else {}

    if llm is not None:
        try:
            prompt = _build_summary_prompt(call_state, entity_summary)
            call_state.tool_calls += 1

            completion = llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a careful, concise call summarization assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            summary_text = completion.choices[0].message.content.strip()
            call_state.tool_successes += 1

        except Exception:
            summary_text = None

    # Fallback if no llm or the call failed
    if not summary_text:
        summary_text = _rule_based_summary(call_state)

    call_state.summary = summary_text
    call_state.step_count += 1

    return call_state
