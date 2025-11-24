# graph/a2a.py

"""
A2A (Agent-to-Agent) protocol utilities.

Agents communicate by writing structured messages into CallState.messages.
Each message has:
    - from: name of the sending agent
    - to: name of the receiving agent (or "any")
    - type: message type string, e.g. "frustration_summary"
    - payload: arbitrary JSON-serializable dict
"""

from __future__ import annotations
from typing import Dict, Any, List

from graph.state import CallState


def send_message(
    state: CallState,
    *,
    from_agent: str,
    to_agent: str,
    msg_type: str,
    payload: Dict[str, Any],
) -> None:
    """
    Append a new A2A message to the call state.
    """
    state.messages.append(
        {
            "from": from_agent,
            "to": to_agent,
            "type": msg_type,
            "payload": payload,
        }
    )


def get_messages_for_agent(
    state: CallState,
    *,
    agent_name: str,
    msg_type: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve messages addressed to a given agent (and optional type filter).
    """
    msgs = [
        m
        for m in state.messages
        if m.get("to") in (agent_name, "any")
    ]
    if msg_type is not None:
        msgs = [m for m in msgs if m.get("type") == msg_type]
    return msgs
