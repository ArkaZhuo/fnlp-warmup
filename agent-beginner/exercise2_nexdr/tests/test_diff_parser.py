from __future__ import annotations

import unittest

from exercise2_nexdr.revision.diff_parser import (
    build_unified_diff,
    parse_markdown_changes,
)


class DiffParserTests(unittest.TestCase):
    def test_parse_markdown_changes(self) -> None:
        old = "# A\nold\n\n## B\nold-b\n"
        new = "# A\nnew\n\n## C\nnew-c\n"
        changes = parse_markdown_changes(old, new)
        self.assertTrue(any(c.section == "A" and c.change_type == "modify" for c in changes))
        self.assertTrue(any(c.section == "B" and c.change_type == "delete" for c in changes))
        self.assertTrue(any(c.section == "C" and c.change_type == "add" for c in changes))

    def test_unified_diff_contains_headers(self) -> None:
        diff = build_unified_diff("a\n", "b\n")
        self.assertIn("draft_v1.md", diff)
        self.assertIn("draft_user_edited.md", diff)


if __name__ == "__main__":
    unittest.main()
