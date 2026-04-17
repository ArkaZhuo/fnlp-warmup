"""Markdown report builder for exercise2."""

from __future__ import annotations

from datetime import datetime

from exercise2_nexdr.core.models import DocumentChunk, PaperRecord


def _format_paper_line(paper: PaperRecord, idx: int) -> str:
    year = paper.year if paper.year is not None else "N/A"
    venue = paper.venue or "Unknown venue"
    url = paper.url or ""
    authors = ", ".join(paper.authors[:5]) or "Unknown authors"
    return (
        f"{idx}. **{paper.title}** ({year}, {venue})\n"
        f"   - Authors: {authors}\n"
        f"   - Citations: {paper.citation_count}\n"
        f"   - Score: {paper.final_score:.3f}\n"
        f"   - URL: {url}"
    )


def build_markdown_report(
    *,
    query: str,
    papers: list[PaperRecord],
    retrieved_chunks: list[DocumentChunk],
    summary: str | None = None,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "# Research Report",
        "",
        f"- Query: {query}",
        f"- Generated At: {now}",
        "- Report Format: Markdown only (HTML disabled by design)",
        "",
        "## Executive Summary",
        "",
    ]

    if summary and summary.strip():
        lines.append(summary.strip())
    else:
        lines.append(
            "This report summarizes relevant literature retrieved via Semantic Scholar and"
            " contextual evidence from user-provided files (including PDF/image ingestion)."
        )

    lines.extend(["", "## Semantic Scholar Findings", ""])
    if papers:
        for idx, paper in enumerate(papers, start=1):
            lines.append(_format_paper_line(paper, idx))
            lines.append("")
    else:
        lines.append("No papers were returned from Semantic Scholar.")
        lines.append("")

    lines.extend(["## Evidence From User Inputs", ""])
    if retrieved_chunks:
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            page_info = f", page={chunk.page_no}" if chunk.page_no else ""
            lines.append(
                f"{idx}. Source: `{chunk.source_file}`{page_info}\n"
                f"   - Snippet: {chunk.content[:500]}"
            )
            lines.append("")
    else:
        lines.append("No relevant snippets were retrieved from user inputs.")
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- HTML generation is intentionally disabled in this exercise2 implementation.",
            "- User-edited markdown can be iteratively revised with the revision engine.",
        ]
    )

    return "\n".join(lines).strip() + "\n"
