# graph/state.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import time


@dataclass
class CallState:
    """
    Per-call, in-memory state.
    """
    # identifiers / raw input
    call_id: Optional[str] = None
    raw_transcript: str = ""

    # cleaned + structured text
    cleaned_transcript: str = ""
    utterances: List[str] = field(default_factory=list)

    # extracted structure
    entities: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    sentiment: str = ""
    frustration_timeline: List[Dict[str, Any]] = field(default_factory=list)
    pain_points: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)

    # 🔹 A2A messages: each message is a dict with from/to/type/payload
    messages: List[Dict[str, Any]] = field(default_factory=list)

    # evaluation metrics (filled by supervisor + evaluation agent)
    evaluation: Dict[str, Any] = field(default_factory=dict)

    # tracing / observability
    start_ts: float = field(default_factory=time.time)
    step_count: int = 0
    tool_calls: int = 0
    tool_successes: int = 0


@dataclass
class MemoryState:
    """
    Long-term memory across calls (Memory Bank style).
    """
    pain_point_counts: Dict[str, int] = field(default_factory=dict)
    sentiment_counts: Dict[str, int] = field(default_factory=dict)
    product_issue_counts: Dict[str, int] = field(default_factory=dict)

    total_calls: int = 0
    avg_faithfulness: float = 0.0
    avg_coverage: float = 0.0
    avg_consistency: float = 0.0
