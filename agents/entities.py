# agents/entities.py

from typing import Dict, Any
import json
import re

from graph.state import CallState
from graph.tools import Tools
from graph.a2a import send_message
from typing import Dict, Any, Optional
from typing import Optional
from graph.tools import Tools


# ------------------ Fallback rule-based extractor ------------------ #

PRODUCT_KEYWORDS = [
    "credit card", "debit card", "checking account", "savings account",
    "mobile app", "website", "online banking", "loan", "mortgage"
]

ISSUE_KEYWORDS = [
    "refund", "chargeback", "login", "password", "declined", "blocked",
    "fee", "overcharged", "statement", "transfer", "fraud", "dispute"
]


def rule_based_entities(text: str) -> Dict[str, Any]:
    text_lower = text.lower()

    product = next((kw for kw in PRODUCT_KEYWORDS if kw in text_lower), None)
    issue = next((kw for kw in ISSUE_KEYWORDS if kw in text_lower), None)

    priority = "normal"
    if re.search(r"\b(cancel\b.*account|close my account)", text_lower):
        priority = "high"
    if re.search(r"supervisor|manager|third time|fourth time", text_lower):
        priority = "high"

    return {
        "customer_profile": None,
        "product": product,
        "issue": issue,
        "context": None,
        "priority": priority,
        "other_tags": [],
    }


# ------------------ Entity Agent (LLM + A2A) ------------------ #

def entities_agent(
    call_state: CallState,
  tools: Optional[Tools] = None,
) -> CallState:

    """
    Entity & Context Extraction Agent.

    INPUT:
        - call_state.cleaned_transcript

    OUTPUT:
        - call_state.entities  (dict)
    """
    text = call_state.cleaned_transcript or call_state.raw_transcript
    llm = tools.get_llm() if tools else None

    entities: Dict[str, Any] = {}

    # Preferred path: LLM extraction
    if llm is not None:
        try:
            prompt = f"""
You extract structured context from customer support calls.

Transcript:
\"\"\"{text}\"\"\"

Return a JSON object with keys:
- customer_profile: brief description, if available
- product: main product discussed
- issue: main problem
- context: important supplementary context
- priority: one of ['low','normal','high']
- other_tags: list of keywords

Your entire answer MUST be valid JSON.
""".strip()

            call_state.tool_calls += 1

            completion = llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            raw = completion.choices[0].message.content.strip()
            call_state.tool_successes += 1

            try:
                entities = json.loads(raw)
            except Exception:
                # LLM returned plain text — attempt heuristic fallback
                entities = rule_based_entities(text)

        except Exception:
            entities = rule_based_entities(text)

    else:
        # No LLM: fallback version
        entities = rule_based_entities(text)

    # ---------- Sanity check ----------
    if not isinstance(entities, dict):
        entities = {"raw_entities": entities}

    call_state.entities = entities

    # ---------- A2A protocol: send entity summary to downstream agents ----------
    payload = {
        "product": entities.get("product"),
        "issue": entities.get("issue"),
        "priority": entities.get("priority"),
        "tags": entities.get("other_tags", []),
    }

    send_message(
        call_state,
        from_agent="entities",
        to_agent="summarization",
        msg_type="entity_summary",
        payload=payload,
    )

    call_state.step_count += 1
    return call_state
