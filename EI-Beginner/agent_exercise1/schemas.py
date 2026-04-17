from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class RetrievalHit:
    source_type: str
    source_name: str
    score: float
    snippet: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CalendarEvent:
    event_id: str
    title: str
    date: str
    start_time: str
    end_time: str | None = None
    location: str | None = None
    participants: list[str] = field(default_factory=list)
    description: str | None = None
    source: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentDecision:
    user_input: str
    intent: str
    missing_fields: list[str]
    retrieved_hits: list[RetrievalHit]
    tool_name: str | None
    tool_args: dict[str, Any]
    response: str


@dataclass
class PendingAction:
    intent: str
    original_user_input: str
    collected_fields: dict[str, Any] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)


@dataclass
class TurnResult:
    decision: AgentDecision
    pending_action: PendingAction | None = None
