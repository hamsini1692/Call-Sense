# agents/actions.py

from typing import List

from graph.state import CallState
from graph.tools import Tools
from typing import Optional


def _rule_based_actions(call_state: CallState) -> List[str]:
    """
    Fallback: simple mapping from pain points & sentiment to actions.
    """
    actions: List[str] = []
    sentiment = call_state.sentiment or "neutral"
    pain_points = call_state.pain_points or []
    text = (call_state.cleaned_transcript or call_state.raw_transcript or "").lower()

    for p in pain_points:
        pl = p.lower()
        if "refund" in pl or "chargeback" in pl:
            actions.append("Check refund/chargeback status and expedite if delayed.")
            actions.append("Provide clear timeline and confirmation email for the refund.")
        elif "login" in pl or "authentication" in pl:
            actions.append("Walk customer through login reset or credential recovery.")
            actions.append("Check for account lock or security flags and resolve.")
        elif "fee" in pl or "overcharged" in pl:
            actions.append("Review recent charges and waive fees if misapplied.")
        elif "repeated unresolved contacts" in pl:
            actions.append("Escalate to Tier-2 support with full case context.")
        else:
            actions.append(f"Investigate and follow standard playbook for: {p}")

    # Generic escalation if sentiment is strongly negative
    if sentiment in ["very_negative", "negative"]:
        actions.append("Offer apology and consider goodwill credit or compensation.")

    # Deduplicate actions
    seen = set()
    deduped = []
    for a in actions:
        if a not in seen:
            seen.add(a)
            deduped.append(a)

    if not deduped:
        deduped = ["Follow standard support procedure and update CRM with call summary."]

    return deduped


def _build_actions_prompt(call_state: CallState) -> str:
    """
    Prompt for LLM to propose targeted, practical actions.
    """
    summary = call_state.summary
    sentiment = call_state.sentiment
    pain_points = call_state.pain_points or []
    entities = call_state.entities or {}

    return f"""
You are an expert customer support coach.

Based on the call information below, propose concrete next best actions
for the agent or operations team.

Summary:
\"\"\"{summary}\"\"\"

Sentiment label: {sentiment}

Pain points:
{pain_points}

Entities/context:
{entities}

Guidelines:
- Suggest 3–6 specific, actionable steps.
- Focus on steps that resolve the customer's issue and prevent future repeat calls.
- Include escalation only when necessary.
- Keep each action as one clear sentence.

Return a JSON array of strings, e.g.:
[
  "Expedite refund processing and send confirmation email to the customer.",
  "Open a ticket with the mobile app team to investigate repeated login failures."
]
""".strip()


def actions_agent(
    call_state: CallState,
    tools: Optional[Tools] = None,
) -> CallState:
    """
    Action Recommendation Agent.

    INPUT:
        - call_state.pain_points
        - call_state.sentiment
        - call_state.summary
        - call_state.entities

    OUTPUT:
        - call_state.recommended_actions: List[str]
    """
    actions: List[str] | None = None
    llm = tools.get_llm() if tools is not None else None

    if llm is not None:
        try:
            prompt = _build_actions_prompt(call_state)
            call_state.tool_calls += 1

            completion = llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You recommend concrete, practical customer support actions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
            )

            content = completion.choices[0].message.content.strip()
            call_state.tool_successes += 1

            import json

            parsed = json.loads(content)
            if isinstance(parsed, list):
                actions = [str(a).strip() for a in parsed if str(a).strip()]

        except Exception:
            actions = None

    if not actions:
        actions = _rule_based_actions(call_state)

    call_state.recommended_actions = actions
    call_state.step_count += 1

    return call_state
