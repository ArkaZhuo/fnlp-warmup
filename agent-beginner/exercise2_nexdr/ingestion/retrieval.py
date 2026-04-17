"""Simple lexical retrieval over ingested chunks."""

from __future__ import annotations

import math
import re
from collections import Counter

from exercise2_nexdr.core.models import DocumentChunk

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")


def _tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_PATTERN.findall(text)]


def retrieve_top_chunks(
    query: str,
    chunks: list[DocumentChunk],
    top_k: int = 6,
) -> list[DocumentChunk]:
    if not query.strip() or not chunks:
        return []

    q_tokens = _tokenize(query)
    if not q_tokens:
        return chunks[: max(1, top_k)]

    q_count = Counter(q_tokens)
    scored: list[tuple[float, DocumentChunk]] = []

    for chunk in chunks:
        c_tokens = _tokenize(chunk.content)
        if not c_tokens:
            continue
        c_count = Counter(c_tokens)

        dot = 0.0
        for token, qv in q_count.items():
            dot += qv * c_count.get(token, 0)

        if dot <= 0:
            continue

        q_norm = math.sqrt(sum(v * v for v in q_count.values()))
        c_norm = math.sqrt(sum(v * v for v in c_count.values()))
        score = dot / max(q_norm * c_norm, 1e-9)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        return [chunk for _, chunk in scored[: max(1, top_k)]]
    return chunks[: max(1, top_k)]
