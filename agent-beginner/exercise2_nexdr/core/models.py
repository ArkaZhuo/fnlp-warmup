"""Shared models for exercise2 NexDR modifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PaperRecord:
    paper_id: str
    title: str
    abstract: str
    year: int | None
    venue: str | None
    url: str | None
    citation_count: int
    authors: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    recency_score: float = 0.0
    citation_score: float = 0.0
    source_quality_score: float = 1.0
    final_score: float = 0.0


@dataclass
class MarkdownSectionChange:
    section: str
    change_type: str
    before: str
    after: str


@dataclass
class DocumentChunk:
    content: str
    source_file: str
    chunk_id: str
    page_no: int | None = None
    bbox: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
