from __future__ import annotations

import unittest

from exercise2_nexdr.core.models import PaperRecord
from exercise2_nexdr.search.semantic_scholar import rank_papers


class SemanticScholarRankTests(unittest.TestCase):
    def test_rank_papers(self) -> None:
        papers = [
            PaperRecord(
                paper_id="1",
                title="Old low citation",
                abstract="",
                year=2010,
                venue=None,
                url=None,
                citation_count=2,
            ),
            PaperRecord(
                paper_id="2",
                title="Recent high citation",
                abstract="",
                year=2025,
                venue=None,
                url=None,
                citation_count=1000,
            ),
        ]
        ranked = rank_papers(papers)
        self.assertEqual(ranked[0].paper_id, "2")


if __name__ == "__main__":
    unittest.main()
