"""Exercise2 runner: NexDR modification demo.

Features:
1) Semantic Scholar search (replace web search)
2) Markdown-only report flow (no html output)
3) User-edit-aware iterative markdown revision
4) Multimodal input ingestion (pdf/image/text)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from exercise2_nexdr.core.report_builder import build_markdown_report
from exercise2_nexdr.ingestion.multimodal_ingestor import ingest_many
from exercise2_nexdr.ingestion.retrieval import retrieve_top_chunks
from exercise2_nexdr.revision.revision_engine import revise_markdown_files
from exercise2_nexdr.search.search_router import search


def _parse_inputs(raw_inputs: list[str] | None) -> list[str]:
    if not raw_inputs:
        return []
    out: list[str] = []
    for raw in raw_inputs:
        parts = [x.strip() for x in raw.split(",") if x.strip()]
        out.extend(parts)
    return out


def run_pipeline(
    *,
    query: str,
    output_dir: str,
    num_results: int,
    input_files: list[str],
    edited_markdown: str | None,
    user_instruction: str | None,
) -> dict:
    workspace = Path(output_dir).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    search_error: str | None = None
    try:
        papers = search(
            query=query,
            search_source="semantic_scholar",
            num_results=num_results,
        )
    except Exception as exc:
        papers = []
        search_error = str(exc)

    ingested_chunks = ingest_many(input_files) if input_files else []
    retrieved_chunks = retrieve_top_chunks(query=query, chunks=ingested_chunks, top_k=8)

    report_v1 = build_markdown_report(
        query=query,
        papers=papers,
        retrieved_chunks=retrieved_chunks,
    )
    report_v1_path = workspace / "markdown_report_v1.md"
    report_v1_path.write_text(report_v1, encoding="utf-8")

    outputs: dict[str, str] = {
        "markdown_report_v1": str(report_v1_path),
    }

    if edited_markdown and user_instruction:
        report_v2_path = workspace / "markdown_report_v2.md"
        revise_markdown_files(
            user_instruction=user_instruction,
            old_markdown_path=str(report_v1_path),
            edited_markdown_path=edited_markdown,
            output_path=str(report_v2_path),
        )
        outputs["markdown_report_v2"] = str(report_v2_path)

    metadata = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "search_source": "semantic_scholar",
        "report_format": "markdown",
        "html_generation": False,
        "paper_count": len(papers),
        "search_error": search_error,
        "input_file_count": len(input_files),
        "ingested_chunk_count": len(ingested_chunks),
        "retrieved_chunk_count": len(retrieved_chunks),
        "outputs": outputs,
    }

    meta_path = workspace / "run_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["metadata_path"] = str(meta_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise2 NexDR modification runner")
    parser.add_argument("--query", required=True, help="Research query")
    parser.add_argument(
        "--output_dir",
        default=f"workspaces/exercise2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--num_results", type=int, default=10)
    parser.add_argument(
        "--inputs",
        nargs="*",
        help="Input files for multimodal ingestion. Supports csv style in one arg or multiple args.",
    )
    parser.add_argument(
        "--edited_markdown",
        help="Path to user-edited markdown for iterative revision.",
    )
    parser.add_argument(
        "--user_instruction",
        help="Additional instruction to apply after user edits.",
    )

    args = parser.parse_args()
    input_files = _parse_inputs(args.inputs)

    result = run_pipeline(
        query=args.query,
        output_dir=args.output_dir,
        num_results=args.num_results,
        input_files=input_files,
        edited_markdown=args.edited_markdown,
        user_instruction=args.user_instruction,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
