"""Revision engine for markdown human-in-the-loop iteration."""

from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI

from exercise2_nexdr.revision.diff_parser import (
    build_unified_diff,
    parse_markdown_changes,
)


def _format_changes_for_prompt(old_markdown: str, edited_markdown: str) -> str:
    changes = parse_markdown_changes(old_markdown, edited_markdown)
    if not changes:
        return "No section-level changes detected."

    blocks: list[str] = []
    for idx, change in enumerate(changes, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{idx}] section={change.section}",
                    f"type={change.change_type}",
                    "before:",
                    change.before[:1500],
                    "after:",
                    change.after[:1500],
                ]
            )
        )
    return "\n\n".join(blocks)


def build_revision_prompt(
    user_instruction: str,
    old_markdown: str,
    edited_markdown: str,
) -> str:
    section_changes = _format_changes_for_prompt(old_markdown, edited_markdown)
    unified_diff = build_unified_diff(old_markdown, edited_markdown)

    return f"""
You are revising a research markdown report.

Rules:
1. Keep output strictly in markdown format.
2. Respect user edits as the latest intent baseline.
3. Apply user's new instruction on top of the edited markdown.
4. Make minimal changes beyond requested edits.
5. Preserve citation markers like [1], [2], etc.

User instruction:
{user_instruction}

Section-level changes detected from user edits:
{section_changes}

Unified diff between previous draft and user-edited draft:
{unified_diff[:6000]}

Current edited markdown baseline (must start from this):
{edited_markdown}

Return only the final revised markdown.
""".strip()


def revise_markdown(
    *,
    user_instruction: str,
    old_markdown: str,
    edited_markdown: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """Revise markdown based on user edits and extra instruction.

    Falls back to returning edited markdown when LLM credentials are missing.
    """

    llm_model = model or os.getenv("LLM_MODEL")
    llm_base_url = base_url or os.getenv("LLM_BASE_URL")
    llm_api_key = api_key or os.getenv("LLM_API_KEY")

    if not (llm_model and llm_base_url and llm_api_key):
        # No runtime llm config -> still finish pipeline deterministically.
        return edited_markdown

    prompt = build_revision_prompt(user_instruction, old_markdown, edited_markdown)
    client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)
    response = client.chat.completions.create(
        model=llm_model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "You are a precise markdown revision assistant.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    return content or edited_markdown


def revise_markdown_files(
    *,
    user_instruction: str,
    old_markdown_path: str,
    edited_markdown_path: str,
    output_path: str,
) -> str:
    old_text = Path(old_markdown_path).read_text(encoding="utf-8", errors="replace")
    edited_text = Path(edited_markdown_path).read_text(
        encoding="utf-8", errors="replace"
    )

    revised = revise_markdown(
        user_instruction=user_instruction,
        old_markdown=old_text,
        edited_markdown=edited_text,
    )
    Path(output_path).write_text(revised, encoding="utf-8")
    return output_path
