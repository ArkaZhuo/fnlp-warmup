from __future__ import annotations

import json
import uuid
from pathlib import Path

from schemas import CalendarEvent


class CalendarStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def _load(self) -> list[dict]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, rows: list[dict]) -> None:
        self.path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_events(self) -> list[CalendarEvent]:
        return [CalendarEvent(**row) for row in self._load()]

    def add_event(self, event: CalendarEvent) -> CalendarEvent:
        rows = self._load()
        rows.append(event.to_dict())
        self._save(rows)
        return event

    def upcoming_events(self, limit: int = 12) -> list[CalendarEvent]:
        events = self.list_events()
        events.sort(key=lambda event: (event.date, event.start_time))
        return events[:limit]

    def find_event(self, keyword: str) -> CalendarEvent | None:
        keyword = keyword.strip()
        for event in self.list_events():
            haystack = " ".join(
                [
                    event.title,
                    event.location or "",
                    " ".join(event.participants),
                    event.description or "",
                ]
            )
            if keyword and keyword in haystack:
                return event
        return None

    def modify_event_time(self, event_id: str, new_start_time: str) -> CalendarEvent | None:
        rows = self._load()
        updated = None
        for row in rows:
            if row["event_id"] == event_id:
                row["start_time"] = new_start_time
                updated = CalendarEvent(**row)
                break
        if updated is not None:
            self._save(rows)
        return updated


def make_event_id() -> str:
    return "evt_" + uuid.uuid4().hex[:10]
