from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from exercise1_nexau import scheduler_tools as st


class SchedulerToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmpdir.name) / "schedule.db")
        st.init_db(db_path=self.db_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    @staticmethod
    def _payload(resp: dict[str, str]) -> dict:
        return json.loads(resp["content"])

    def test_create_query_update_delete_flow(self) -> None:
        create = self._payload(
            st.schedule_create(
                title="项目复盘",
                start_time="2026-04-10 10:00",
                end_time="2026-04-10 11:00",
                timezone="Asia/Shanghai",
                db_path=self.db_path,
            )
        )
        self.assertTrue(create["ok"])
        event_id = create["event"]["id"]

        query = self._payload(st.schedule_query(keyword="复盘", db_path=self.db_path))
        self.assertTrue(query["ok"])
        self.assertEqual(query["count"], 1)

        updated = self._payload(
            st.schedule_update(
                event_id=event_id,
                start_time="2026-04-10 11:00",
                end_time="2026-04-10 12:00",
                db_path=self.db_path,
            )
        )
        self.assertTrue(updated["ok"])
        self.assertIn("11:00", updated["event"]["start_time"])

        delete_without_confirm = self._payload(st.schedule_delete(event_id=event_id, db_path=self.db_path))
        self.assertFalse(delete_without_confirm["ok"])
        self.assertEqual(delete_without_confirm["error"], "confirmation_required")

        delete_confirmed = self._payload(st.schedule_delete(event_id=event_id, confirm=True, db_path=self.db_path))
        self.assertTrue(delete_confirmed["ok"])
        self.assertEqual(delete_confirmed["event"]["status"], "cancelled")

    def test_ingest_markdown_upsert(self) -> None:
        md1 = Path(self.tmpdir.name) / "notes_v1.md"
        md2 = Path(self.tmpdir.name) / "notes_v2.md"
        md1.write_text("- 2026-05-01 10:00 研发例会\n", encoding="utf-8")
        md2.write_text("- 2026-05-01 11:00 研发例会\n", encoding="utf-8")

        first_ingest = self._payload(
            st.ingest_markdown_schedules(file_path=str(md1), mode="create", db_path=self.db_path)
        )
        self.assertTrue(first_ingest["ok"])
        self.assertEqual(first_ingest["created_count"], 1)

        second_ingest = self._payload(
            st.ingest_markdown_schedules(file_path=str(md2), mode="upsert", db_path=self.db_path)
        )
        self.assertTrue(second_ingest["ok"])
        self.assertEqual(second_ingest["updated_count"], 1)

        query = self._payload(st.schedule_query(keyword="研发例会", db_path=self.db_path))
        self.assertEqual(query["count"], 1)
        self.assertIn("11:00", query["events"][0]["start_time"])

    def test_ingest_feishu_doc_requires_credentials(self) -> None:
        old_app_id = os.environ.pop("FEISHU_APP_ID", None)
        old_app_secret = os.environ.pop("FEISHU_APP_SECRET", None)
        try:
            result = self._payload(
                st.ingest_feishu_doc_schedules(
                    document_id_or_url="doxcn123",
                    db_path=self.db_path,
                )
            )
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"], "missing_feishu_credentials")
        finally:
            if old_app_id is not None:
                os.environ["FEISHU_APP_ID"] = old_app_id
            if old_app_secret is not None:
                os.environ["FEISHU_APP_SECRET"] = old_app_secret

    def test_extract_feishu_resource(self) -> None:
        kind, token = st._extract_feishu_resource("https://xxx.feishu.cn/docx/doxcnABC123")
        self.assertEqual(kind, "docx")
        self.assertEqual(token, "doxcnABC123")

        kind, token = st._extract_feishu_resource("https://xxx.feishu.cn/wiki/GpwnwG2UQiZ0bukNKDLcqoHvntd")
        self.assertEqual(kind, "wiki")
        self.assertEqual(token, "GpwnwG2UQiZ0bukNKDLcqoHvntd")

        kind, token = st._extract_feishu_resource("doxcnABC123")
        self.assertEqual(kind, "docx")
        self.assertEqual(token, "doxcnABC123")

    def test_extract_titles_from_markdown_and_chat_are_clean(self) -> None:
        markdown_candidate = st._extract_event_from_line(
            "- 2026-04-10 10:00-11:00 项目复盘会议",
            line_no=1,
            timezone="Asia/Shanghai",
            default_duration_minutes=60,
        )
        self.assertIsNotNone(markdown_candidate)
        self.assertEqual(markdown_candidate["title"], "项目复盘会议")

        chat_candidate = st._extract_event_from_line(
            "[2026-04-09 18:02] 张三: 咱们 2026-04-13 10:00 开站会",
            line_no=1,
            timezone="Asia/Shanghai",
            default_duration_minutes=60,
            strip_chat_prefix=True,
        )
        self.assertIsNotNone(chat_candidate)
        self.assertEqual(chat_candidate["title"], "开站会")


if __name__ == "__main__":
    unittest.main()
