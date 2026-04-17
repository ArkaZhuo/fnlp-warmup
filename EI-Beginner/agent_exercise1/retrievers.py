from __future__ import annotations

import re
from pathlib import Path

from schemas import RetrievalHit


TIME_KEYWORDS = ["明天", "后天", "下午", "上午", "晚上", "周", "14:00", "15:00", "16:00", "两点", "三点"]
TOPIC_KEYWORDS = ["论文", "开题", "组会", "实验", "周报", "讨论", "会议室", "光华楼", "逸夫楼", "导师", "老师"]


def tokenize(text: str) -> list[str]:
    parts = re.split(r"[\s,，。；;：:\-_/（）()\[\]]+", text)
    tokens = [p for p in parts if p]
    people = re.findall(r"([A-Za-z0-9\u4e00-\u9fff]{1,6}(?:老师|导师|同学))", text)
    tokens.extend(people)
    for kw in TOPIC_KEYWORDS:
        if kw in text:
            tokens.append(kw)
    return list(dict.fromkeys(tokens))


def score_text(query: str, text: str) -> float:
    query_tokens = tokenize(query)
    score = 0.0
    for token in query_tokens:
        if token in text:
            score += 2.0 if len(token) >= 2 else 0.5
    for kw in TIME_KEYWORDS:
        if kw in query and kw in text:
            score += 0.8
    return score


class TextCorpusRetriever:
    def __init__(self, root: Path, source_type: str):
        self.root = root
        self.source_type = source_type

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        if not self.root.exists():
            return hits

        for path in sorted(self.root.glob("**/*")):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            score = score_text(query, text)
            if score <= 0:
                continue

            best_line = ""
            best_line_score = -1.0
            for line in text.splitlines():
                line_score = score_text(query, line)
                if line_score > best_line_score:
                    best_line_score = line_score
                    best_line = line.strip()

            hits.append(
                RetrievalHit(
                    source_type=self.source_type,
                    source_name=path.name,
                    score=score + max(best_line_score, 0.0),
                    snippet=best_line or text[:120],
                    metadata={"path": str(path)},
                )
            )

        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[:top_k]


class MultiSourceRetriever:
    def __init__(self, base_dir: Path):
        self.chat = TextCorpusRetriever(base_dir / "chat_history", "chat_history")
        self.notes = TextCorpusRetriever(base_dir / "notes", "markdown_note")
        self.feishu = TextCorpusRetriever(base_dir / "feishu", "feishu_doc")

    def search_all(self, query: str, top_k_per_source: int = 3) -> list[RetrievalHit]:
        hits = []
        hits.extend(self.chat.search(query, top_k=top_k_per_source))
        hits.extend(self.notes.search(query, top_k=top_k_per_source))
        hits.extend(self.feishu.search(query, top_k=top_k_per_source))
        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[:6]

    def search_multi_query(self, queries: list[str], top_k_per_query: int = 3) -> list[RetrievalHit]:
        merged: dict[tuple[str, str], RetrievalHit] = {}
        for query in queries:
            if not query.strip():
                continue
            for hit in self.search_all(query, top_k_per_source=top_k_per_query):
                key = (hit.source_type, hit.source_name)
                if key not in merged or hit.score > merged[key].score:
                    merged[key] = hit
        hits = list(merged.values())
        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[:8]
