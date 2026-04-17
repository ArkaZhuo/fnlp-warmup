"""Search router for exercise2.

NexDR exercise2 requirement: replace generic web search with Semantic Scholar.
"""

from __future__ import annotations

from typing import Any

from exercise2_nexdr.core.models import PaperRecord
from exercise2_nexdr.search.semantic_scholar import search_semantic_scholar


def search(
    query: str,
    search_source: str = "semantic_scholar",
    num_results: int = 10,
    **kwargs: Any,
) -> list[PaperRecord]:
    source = (search_source or "semantic_scholar").strip().lower()
    if source not in {"semantic_scholar", "s2", "web"}:
        raise ValueError(
            f"Unsupported search_source={search_source}. Use semantic_scholar."
        )

    # Keep backward compatibility with old 'web' call path by redirecting it.
    return search_semantic_scholar(
        query=query,
        limit=num_results,
        year_from=kwargs.get("year_from"),
        year_to=kwargs.get("year_to"),
    )
