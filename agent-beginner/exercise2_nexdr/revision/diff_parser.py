"""Markdown diff parser for user-edited report iteration."""

from __future__ import annotations

import difflib
import re
from collections import OrderedDict

from exercise2_nexdr.core.models import MarkdownSectionChange

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")


def split_markdown_sections(markdown_text: str) -> OrderedDict[str, str]:
    """Split markdown into heading-based sections.

    Returns ordered mapping: section_name -> content.
    A pseudo section '__PREFACE__' stores content before first heading.
    """

    lines = markdown_text.splitlines()
    sections: OrderedDict[str, list[str]] = OrderedDict()
    current = "__PREFACE__"
    sections[current] = []

    for line in lines:
        match = HEADING_PATTERN.match(line)
        if match:
            current = match.group(2).strip() or "__UNTITLED__"
            sections.setdefault(current, [])
        sections[current].append(line)

    return OrderedDict((name, "\n".join(content).strip()) for name, content in sections.items())


def parse_markdown_changes(old_markdown: str, new_markdown: str) -> list[MarkdownSectionChange]:
    old_sections = split_markdown_sections(old_markdown)
    new_sections = split_markdown_sections(new_markdown)

    changes: list[MarkdownSectionChange] = []

    old_keys = set(old_sections.keys())
    new_keys = set(new_sections.keys())

    for removed in sorted(old_keys - new_keys):
        changes.append(
            MarkdownSectionChange(
                section=removed,
                change_type="delete",
                before=old_sections[removed],
                after="",
            )
        )

    for added in sorted(new_keys - old_keys):
        changes.append(
            MarkdownSectionChange(
                section=added,
                change_type="add",
                before="",
                after=new_sections[added],
            )
        )

    for shared in old_keys & new_keys:
        before = old_sections[shared]
        after = new_sections[shared]
        if before != after:
            changes.append(
                MarkdownSectionChange(
                    section=shared,
                    change_type="modify",
                    before=before,
                    after=after,
                )
            )

    # Keep stable ordering aligned with new markdown structure
    order_map = {name: idx for idx, name in enumerate(new_sections.keys())}
    changes.sort(key=lambda c: order_map.get(c.section, 10**9))
    return changes


def build_unified_diff(old_markdown: str, new_markdown: str) -> str:
    old_lines = old_markdown.splitlines(keepends=True)
    new_lines = new_markdown.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile="draft_v1.md",
        tofile="draft_user_edited.md",
        n=3,
    )
    return "".join(diff)
