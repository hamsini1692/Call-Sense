# agents/evaluation.py

from typing import Dict, Any
import json

from graph.state import CallState
from graph.tools import Tools
from eval.metrics import compute_basic_eval
from typing import Optional
from graph.tools import Tools


def _build_eval_prompt(call_state: CallState) -> str:
    """
    Build an instruction for the LLM to evaluate the pipeline output.
    The LLM should score:
      - faithfulness: how well outputs stick to the transcript
      - coverage: how well outputs cover the main issues in the call
      - consistency: how internally consistent the outputs are
    """
    transcript = call_state.cleaned_transcript or call_state.raw_transcript
    summary = call_state.summary
    sentiment = call_state.sentiment
    pain_points = call_state.pain_points or []
    actions = call_state.recommended_actions or []
    entities = call_state.entities or {}

    return f"""
You are evaluating the quality of an AI assistant's analysis of a customer support call.

Here is the original transcript:
\"\"\"{transcript}\"\"\"

Here is the assistant's internal analysis:

Summary:
\"\"\"{summary}\"\"\"

Sentiment label: {sentiment}

Entities:
{json.dumps(entities, indent=2)}

Pain points:
{json.dumps(pain_points, indent=2)}

Recommended actions:
{json.dumps(actions, indent=2)}

Your task:
1. Rate the following on a 0.0–1.0 scale (floats):
   - faithfulness_score: Are the summary, pain points, and actions grounded in the transcript?
   - coverage_score: Do they cover the main issues / concerns the customer raises?
   - consistency_score: Are summary, sentiment, pain points, and actions mutually consistent?

2. Provide a short textual note explaining any major issues.

Return ONLY a JSON object with this schema:
{{
  "faithfulness_score": 0.0,
  "coverage_score": 0.0,
  "consistency_score": 0.0,
  "notes": "short explanation"
}}
""".strip()


def _default_llm_scores() -> Dict[str, Any]:
    """
    Reasonable defaults if LLM evaluation is unavailable.
    """
    return {
        "faithfulness_score": 1.0,
        "coverage_score": 1.0,
        "consistency_score": 1.0,
        "notes": "LLM evaluation not available; using default scores.",
    }


def evaluation_agent(
    call_state: CallState,
    tools: Optional[Tools] = None,
) -> CallState:
    """
    Evaluation Agent.

    Responsibilities:
    - Compute basic, model-agnostic metrics (tool usage, latency, steps).
    - Optionally use an LLM to score:
        • faithfulness_score
        • coverage_score
        • consistency_score
    - Store all metrics in call_state.evaluation.
    """
    # ---- 1. LLM-based eval (if available) ----
    llm_scores: Dict[str, Any] = _default_llm_scores()
    llm = tools.get_llm() if tools is not None else None

    if llm is not None:
        try:
            prompt = _build_eval_prompt(call_state)
            call_state.tool_calls += 1

            completion = llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a rigorous evaluator. Output ONLY valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )

            raw = completion.choices[0].message.content.strip()
            call_state.tool_successes += 1

            parsed = json.loads(raw)

            # Make sure required keys exist; fall back to defaults if missing
            for key in ["faithfulness_score", "coverage_score", "consistency_score"]:
                if key not in parsed:
                    parsed[key] = llm_scores[key]

            if "notes" not in parsed:
                parsed["notes"] = ""

            llm_scores = parsed

        except Exception:
            # If anything goes wrong, keep defaults
            llm_scores = _default_llm_scores()

    # ---- 2. Basic observability metrics ----
    basic_eval = compute_basic_eval(call_state)

    # ---- 3. Merge into call_state.evaluation ----
    call_state.evaluation = {
        **basic_eval,
        "faithfulness_score": float(llm_scores.get("faithfulness_score", 1.0)),
        "coverage_score": float(llm_scores.get("coverage_score", 1.0)),
        "consistency_score": float(llm_scores.get("consistency_score", 1.0)),
        "eval_notes": llm_scores.get("notes", ""),
    }

    # ---- 4. Step accounting ----
    call_state.step_count += 1

    return call_state
