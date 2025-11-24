# eval/metrics.py

"""
Metrics utilities for the CallSense multi-agent system.

This module is responsible for:
- Computing per-call evaluation metrics from CallState
- Updating long-term MemoryState aggregates across calls

It is intentionally LLM-agnostic: any LLM-based checks
(e.g., faithfulness scoring) should be done in the evaluation_agent,
which can then store results in call_state.evaluation.
"""

from __future__ import annotations
from typing import Dict, Any
import time

from graph.state import CallState, MemoryState


def compute_basic_eval(call_state: CallState) -> Dict[str, Any]:
    """
    Compute basic, model-agnostic evaluation metrics for a single call.

    These metrics measure *how the agent system behaved*:
    - how many tools were called, and how many succeeded
    - how many agent steps ran
    - how long the pipeline took

    Returns a dict that can be merged into call_state.evaluation.
    """
    tool_calls = call_state.tool_calls
    tool_successes = call_state.tool_successes

    tool_success_rate = (tool_successes / tool_calls) if tool_calls else 1.0

    # duration from when CallState was created until now
    duration_sec = max(0.0, time.time() - call_state.start_ts)

    return {
        "tool_calls": tool_calls,
        "tool_successes": tool_successes,
        "tool_success_rate": tool_success_rate,
        "step_count": call_state.step_count,
        "duration_sec": duration_sec,
    }


def update_memory_from_call(memory: MemoryState, call_state: CallState) -> MemoryState:
    """
    Update long-term MemoryState using information from a single processed call.

    This gives you trends across calls, such as:
    - distribution of sentiment labels
    - most common pain points
    - product-level issue frequencies
    - running averages of eval scores (if present)

    This function mutates and also returns the MemoryState for convenience.
    """
    memory.total_calls += 1
    n = memory.total_calls

    # --- sentiment distribution ---
    sentiment = call_state.sentiment or "unknown"
    memory.sentiment_counts[sentiment] = memory.sentiment_counts.get(sentiment, 0) + 1

    # --- pain point frequencies ---
    for p in call_state.pain_points or []:
        memory.pain_point_counts[p] = memory.pain_point_counts.get(p, 0) + 1

    # --- product-level issue counts (if entities contains "product") ---
    entities = call_state.entities or {}
    product = entities.get("product")
    if product:
        memory.product_issue_counts[product] = memory.product_issue_counts.get(product, 0) + 1

    # --- running averages for eval scores, if present ---
    eval_data = call_state.evaluation or {}

    # helper: incremental running average
    def _update_running_avg(current_avg: float, new_value: float, count: int) -> float:
        return current_avg + (new_value - current_avg) / float(count)

    if "faithfulness_score" in eval_data:
        memory.avg_faithfulness = _update_running_avg(
            memory.avg_faithfulness, float(eval_data["faithfulness_score"]), n
        )

    if "coverage_score" in eval_data:
        memory.avg_coverage = _update_running_avg(
            memory.avg_coverage, float(eval_data["coverage_score"]), n
        )

    if "consistency_score" in eval_data:
        memory.avg_consistency = _update_running_avg(
            memory.avg_consistency, float(eval_data["consistency_score"]), n
        )

    return memory
