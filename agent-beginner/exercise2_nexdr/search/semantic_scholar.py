"""Semantic Scholar search provider for exercise2.

This module replaces generic web search with Semantic Scholar API.
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Any

import requests

from exercise2_nexdr.core.models import PaperRecord

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
DEFAULT_FIELDS = "paperId,title,abstract,year,venue,url,citationCount,authors"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_citation(citation_count: int, max_seen: int) -> float:
    if citation_count <= 0 or max_seen <= 0:
        return 0.0
    return min(math.log1p(citation_count) / math.log1p(max_seen), 1.0)


def _normalize_recency(year: int | None, current_year: int) -> float:
    if year is None:
        return 0.0
    # Last 10 years receives most weight.
    age = max(current_year - year, 0)
    return max(0.0, 1.0 - age / 10.0)


def _build_paper(item: dict[str, Any]) -> PaperRecord:
    return PaperRecord(
        paper_id=str(item.get("paperId") or ""),
        title=str(item.get("title") or "").strip(),
        abstract=str(item.get("abstract") or "").strip(),
        year=_safe_int(item.get("year"), default=0) or None,
        venue=(str(item.get("venue") or "").strip() or None),
        url=(str(item.get("url") or "").strip() or None),
        citation_count=max(0, _safe_int(item.get("citationCount"), default=0)),
        authors=[
            str(author.get("name") or "").strip()
            for author in (item.get("authors") or [])
            if str(author.get("name") or "").strip()
        ],
    )


def rank_papers(papers: list[PaperRecord]) -> list[PaperRecord]:
    """Rank papers with a weighted score.

    score = 0.45 * relevance + 0.25 * recency + 0.20 * citation + 0.10 * source_quality
    """

    if not papers:
        return []

    current_year = datetime.now().year
    max_citation = max(p.citation_count for p in papers) if papers else 0

    for idx, paper in enumerate(papers):
        # Semantic Scholar API returns relevance-ordered results.
        # Use descending rank position as relevance proxy.
        paper.relevance_score = 1.0 - (idx / max(len(papers), 1))
        paper.recency_score = _normalize_recency(paper.year, current_year)
        paper.citation_score = _normalize_citation(paper.citation_count, max_citation)
        paper.source_quality_score = 1.0
        paper.final_score = (
            0.45 * paper.relevance_score
            + 0.25 * paper.recency_score
            + 0.20 * paper.citation_score
            + 0.10 * paper.source_quality_score
        )

    return sorted(papers, key=lambda p: p.final_score, reverse=True)


def search_semantic_scholar(
    query: str,
    limit: int = 10,
    year_from: int | None = None,
    year_to: int | None = None,
    timeout: float = 30.0,
) -> list[PaperRecord]:
    """Search papers from Semantic Scholar Graph API."""

    if not query.strip():
        raise ValueError("query is required")

    safe_limit = max(1, min(limit, 100))
    params: dict[str, Any] = {
        "query": query,
        "limit": safe_limit,
        "fields": DEFAULT_FIELDS,
    }

    # Semantic Scholar supports year filter as "YYYY-" / "YYYY-YYYY"
    if year_from is not None and year_to is not None:
        params["year"] = f"{year_from}-{year_to}"
    elif year_from is not None:
        params["year"] = f"{year_from}-"

    headers = {"User-Agent": "exercise2-nexdr/1.0"}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    try:
        response = requests.get(
            SEMANTIC_SCHOLAR_API_URL,
            params=params,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status in {429, 500, 502, 503, 504}:
            # Keep pipeline alive under transient API throttling.
            return []
        raise
    except requests.RequestException:
        # For offline/dev environments, return empty and allow downstream flow.
        return []

    payload = response.json()
    raw_items = payload.get("data") or []
    papers = [_build_paper(item) for item in raw_items]
    papers = [paper for paper in papers if paper.title]
    return rank_papers(papers)
