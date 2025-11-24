# graph/supervisor.py

"""
Supervisor orchestrates the multi-agent CallSense pipeline.

- Initializes CallState
- Applies agents in sequence
- Passes shared Tools container (LLM, MCP cleaner, data loader)
- Supports A2A message protocol through CallState.messages
- Tracks MemoryState for long-term trends (optional)
"""

from __future__ import annotations
from typing import Optional

from graph.state import CallState, MemoryState
from graph.tools import Tools, default_tools
from graph.agents import AGENT_EXECUTION_ORDER, get_agent
from typing import Optional


# Optional global memory (can also be stored to disk)
GLOBAL_MEMORY = MemoryState()


def run_pipeline(
    raw_transcript: str,
    llm_client,
    cleaner=None,
    data_loader=None,
    update_memory: bool = True,
) -> CallState:
    """
    Run the full CallSense multi-agent pipeline.

    Args:
        raw_transcript: raw text from CSV / UI
        llm_client: OpenAI client (or similar)
        cleaner: MCP or custom cleaning tool (optional)
        data_loader: CSV loader (optional)
        update_memory: whether to update long-term MemoryState

    Returns:
        Final CallState with all agent outputs and evaluation results.
    """

    # ---- 1. Build Tools container ----
    tools = default_tools(
        llm_client=llm_client,
        cleaner=cleaner,
        data_loader=data_loader,
    )

    # ---- 2. Initialize per-call state ----
    call_state = CallState(raw_transcript=raw_transcript)

    # ---- 3. Execute agents in the canonical order (A2A-supported) ----
    for agent_name in AGENT_EXECUTION_ORDER:
        agent_fn = get_agent(agent_name)
        call_state = agent_fn(call_state, tools=tools)

    # ---- 4. Evaluation agent (optional if registered separately) ----
    # If evaluation agent is in registry, you can include it there.
    # Otherwise, you can call it manually here.
    try:
        eval_fn = get_agent("evaluation")
        call_state = eval_fn(call_state, tools=tools)
    except KeyError:
        # evaluation not registered, skip
        pass

    # ---- 5. Update long-term memory ----
    if update_memory:
        from eval.metrics import update_memory_from_call
        update_memory_from_call(GLOBAL_MEMORY, call_state)

    # ---- 6. Return to UI ----
    return call_state
