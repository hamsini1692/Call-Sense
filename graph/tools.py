# graph/tools.py

"""
tools.py — centralized container for all external tools used
by the CallSense multi-agent system.

This includes:
- OpenAI LLM client
- MCP transcript cleaning tool (optional)
- Data loader for CSV/dataset (optional)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Protocol


# ---------- Protocols (interfaces) ---------- #

class TranscriptCleaner(Protocol):
    def clean_transcript(self, text: str) -> str:
        ...


class DataLoader(Protocol):
    def load(self) -> Any:
        ...


# ---------- Tools container ---------- #

@dataclass
class Tools:
    """
    Container for external tools.

    llm:    OpenAI (or compatible) client
    cleaner: transcript cleaning tool (MCP or custom) – optional
    data_loader: loader for your CSV or dataset – optional
    """
    llm: Any
    cleaner: Optional[TranscriptCleaner] = None
    data_loader: Optional[DataLoader] = None

    def get_llm(self) -> Any:
        return self.llm

    def get_cleaner(self) -> Optional[TranscriptCleaner]:
        return self.cleaner

    def get_data_loader(self) -> Optional[DataLoader]:
        return self.data_loader


def default_tools(
    llm_client: Any,
    cleaner: Optional[TranscriptCleaner] = None,
    data_loader: Optional[DataLoader] = None,
) -> Tools:
    """
    Convenience factory for building the Tools bundle.
    """
    return Tools(
        llm=llm_client,
        cleaner=cleaner,
        data_loader=data_loader,
    )
