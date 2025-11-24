# graph/agents.py

"""
Agent registry and execution order for the CallSense multi-agent system.

This module provides:
- A common AgentFn type alias.
- A registry mapping string names → agent callables.
- A default execution order used by the Supervisor.

The Supervisor can:
- Run agents in sequence using AGENT_EXECUTION_ORDER.
- Address agents by name (e.g., for debugging, conditional runs).
"""

from typing import Callable, Dict, List

from graph.state import CallState
from graph.tools import Tools

# Import individual agent implementations
from agents.cleaning import cleaning_agent
from agents.entities import entities_agent
from agents.summarization import summarization_agent
from agents.sentiment import sentiment_agent
from agents.frustration_loop import frustration_loop_agent
from agents.pain_points import pain_points_agent
from agents.actions import actions_agent
from agents.evaluation import evaluation_agent
from typing import Callable, Dict, List, Optional
from graph.tools import Tools


# Type alias for all agents:
# Each agent takes (CallState, Tools|None) and returns an updated CallState.
AgentFn = Callable[[CallState, Optional[Tools]], CallState]


#: Registry mapping agent names → agent functions.
#: The names are what the Supervisor can use to refer to them.
AGENT_REGISTRY: Dict[str, AgentFn] = {
    "cleaning": cleaning_agent,
    "entities": entities_agent,
    "summarization": summarization_agent,
    "sentiment": sentiment_agent,
    "frustration_loop": frustration_loop_agent,
    "pain_points": pain_points_agent,
    "actions": actions_agent,
    "evaluation": evaluation_agent,
}


#: Default linear execution order for a full call analysis
#: (excluding evaluation, which Supervisor may call separately).
AGENT_EXECUTION_ORDER: List[str] = [
    "cleaning",
    "entities",
    "summarization",
    "sentiment",
    "frustration_loop",
    "pain_points",
    "actions",
    # "evaluation" is intentionally omitted here to avoid double-running
]


def get_agent(name: str) -> AgentFn:
    """
    Convenience helper to fetch an agent by name.

    Raises:
        KeyError if the agent name is not registered.
    """
    try:
        return AGENT_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown agent name: {name!r}. "
            f"Known agents: {list(AGENT_REGISTRY.keys())}"
        )
