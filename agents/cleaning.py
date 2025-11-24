# agents/cleaning.py

from typing import List
import re

from graph.state import CallState
from graph.tools import Tools, TranscriptCleaner
from typing import Optional
from graph.tools import Tools


def _fallback_clean(text: str) -> str:
    """
    Basic normalization fallback when MCP/custom cleaner is not available.
    """
    if not text:
        return ""

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove obvious artifacts (optional)
    text = re.sub(r"\[(noise|music|silence)\]", "", text, flags=re.IGNORECASE)

    return text


def _split_utterances(text: str) -> List[str]:
    """
    Very lightweight segmentation.
    In a real system you may have diarization, timestamps, speaker tags, etc.
    """
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def cleaning_agent(
    call_state: CallState,
    tools: Optional[Tools] = None,
) -> CallState:
    ...

    """
    Transcript Cleaning Agent.

    Responsibilities:
    - Use MCP/custom cleaner (preferred)
    - Fallback to basic regex cleaning
    - Populate:
        • call_state.cleaned_transcript
        • call_state.utterances
    - Update observability fields
    """
    raw_text = call_state.raw_transcript or ""
    cleaned = raw_text

    cleaner: TranscriptCleaner | None = tools.get_cleaner() if tools else None

    # 1. Try MCP cleaner if provided
    if cleaner is not None:
        try:
            call_state.tool_calls += 1
            cleaned = cleaner.clean_transcript(raw_text)
            call_state.tool_successes += 1
        except Exception:
            cleaned = _fallback_clean(raw_text)
    else:
        # 2. Fallback cleaning
        cleaned = _fallback_clean(raw_text)

    # 3. Basic segmentation
    call_state.cleaned_transcript = cleaned
    call_state.utterances = _split_utterances(cleaned)

    call_state.step_count += 1
    return call_state
