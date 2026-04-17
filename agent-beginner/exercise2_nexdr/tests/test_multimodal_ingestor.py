from __future__ import annotations

import unittest
from pathlib import Path

from exercise2_nexdr.ingestion.multimodal_ingestor import ingest_path


class MultimodalIngestorTests(unittest.TestCase):
    def test_ingest_markdown(self) -> None:
        path = Path("/home/df_05/A_fnlp/agent-beginner/exercise2_nexdr/samples/sample_notes.md")
        chunks = ingest_path(str(path))
        self.assertTrue(len(chunks) > 0)
        self.assertIn("AI Research Notes", chunks[0].content)

    def test_ingest_pdf(self) -> None:
        path = Path("/home/df_05/A_fnlp/agent-beginner/exercise2_nexdr/samples/sample_pdf.pdf")
        chunks = ingest_path(str(path))
        self.assertTrue(len(chunks) > 0)

    def test_ingest_image(self) -> None:
        path = Path("/home/df_05/A_fnlp/agent-beginner/exercise2_nexdr/samples/sample_image.png")
        chunks = ingest_path(str(path))
        self.assertTrue(len(chunks) > 0)


if __name__ == "__main__":
    unittest.main()
