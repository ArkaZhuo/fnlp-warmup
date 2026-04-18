"""Custom schedule tools for NexAU exercise 1.

This module implements:
- schedule create/update/query/delete tools
- ingestion tools for markdown/chat/feishu-exported-markdown
- sqlite-based storage with conflict detection
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_DURATION_MINUTES = 60
DEFAULT_DB_PATH = Path(__file__).parent / "data" / "schedule.db"
DEFAULT_FEISHU_BASE_URL = "https://open.feishu.cn"

DATETIME_TOKEN_PATTERN = re.compile(
    r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?"
    r"|\d{4}年\d{1,2}月\d{1,2}日\s*\d{1,2}:\d{2}"
    r"|\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}"
    r"|(?:今天|明天|后天)\s*\d{1,2}:\d{2}"
    r"|下周[一二三四五六日天]\s*\d{1,2}:\d{2})",
)

TIME_RANGE_PATTERN = re.compile(r"(\d{1,2}:\d{2})\s*[-~到]\s*(\d{1,2}:\d{2})")
LEADING_FILLER_PATTERN = re.compile(r"^(?:咱们|我们|安排|约一下|约个|请|麻烦|帮我|帮忙)\s+")
BULLET_PREFIX_PATTERN = re.compile(r"^\s*(?:[-*+]\s*(?:\[[xX ]\]\s*)?|\d+[\.)]\s*)")
CHAT_PREFIX_PATTERN = re.compile(
    r"^\s*\[(?P<timestamp>\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)\]\s*"
    r"(?:(?P<speaker>[^:：]{1,30})[:：]\s*)?(?P<body>.*)$"
)

WEEKDAY_MAP = {
    "一": 0,
    "二": 1,
    "三": 2,
    "四": 3,
    "五": 4,
    "六": 5,
    "日": 6,
    "天": 6,
}


@dataclass
class Event:
    id: str
    title: str
    description: str | None
    start_time_utc: str
    end_time_utc: str
    timezone: str
    location: str | None
    source_type: str
    source_ref: str | None
    source_line: int | None
    feishu_calendar_id: str | None
    feishu_event_id: str | None
    feishu_sync_error: str | None
    status: str
    created_at: str
    updated_at: str


def _tool_response(payload: dict[str, Any], return_display: str) -> dict[str, str]:
    return {
        "content": json.dumps(payload, ensure_ascii=False),
        "returnDisplay": return_display,
    }


def _db_path(db_path: str | None) -> Path:
    if db_path:
        return Path(db_path).expanduser().resolve()
    return DEFAULT_DB_PATH.resolve()


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    resolved = _db_path(db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | None = None) -> dict[str, str]:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                start_time_utc TEXT NOT NULL,
                end_time_utc TEXT NOT NULL,
                timezone TEXT NOT NULL,
                location TEXT,
                source_type TEXT NOT NULL DEFAULT 'manual',
                source_ref TEXT,
                source_line INTEGER,
                feishu_calendar_id TEXT,
                feishu_event_id TEXT,
                feishu_sync_error TEXT,
                status TEXT NOT NULL DEFAULT 'confirmed',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
        )
        existing_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(events)").fetchall()
        }
        migrations = {
            "feishu_calendar_id": "ALTER TABLE events ADD COLUMN feishu_calendar_id TEXT",
            "feishu_event_id": "ALTER TABLE events ADD COLUMN feishu_event_id TEXT",
            "feishu_sync_error": "ALTER TABLE events ADD COLUMN feishu_sync_error TEXT",
        }
        for column, sql in migrations.items():
            if column not in existing_columns:
                conn.execute(sql)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_start ON events(start_time_utc)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_title ON events(title)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)")
    return _tool_response({"ok": True, "db_path": str(_db_path(db_path))}, "Schedule database ready")


def _now_utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _normalize_timezone(timezone: str | None) -> str:
    tz = timezone or DEFAULT_TIMEZONE
    try:
        ZoneInfo(tz)
    except Exception as exc:
        raise ValueError(f"Invalid timezone: {tz}") from exc
    return tz


def _to_utc_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).replace(microsecond=0).isoformat()


def _parse_datetime(
    value: str,
    timezone: str = DEFAULT_TIMEZONE,
    reference_dt: datetime | None = None,
) -> datetime:
    if not value or not value.strip():
        raise ValueError("datetime value is required")

    tz_name = _normalize_timezone(timezone)
    tz = ZoneInfo(tz_name)
    raw = value.strip()

    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"

    # ISO path
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        return dt
    except ValueError:
        pass

    # Relative shortcuts: 今天/明天/后天 HH:MM
    rel_match = re.match(r"^(今天|明天|后天)\s*(\d{1,2}):(\d{2})$", raw)
    if rel_match:
        base = (reference_dt.astimezone(tz) if reference_dt else datetime.now(tz)).replace(
            second=0,
            microsecond=0,
        )
        day_token, hour_s, minute_s = rel_match.groups()
        offset = {"今天": 0, "明天": 1, "后天": 2}[day_token]
        return (base + timedelta(days=offset)).replace(hour=int(hour_s), minute=int(minute_s))

    # Relative shortcuts: 下周X HH:MM
    next_week_match = re.match(r"^下周([一二三四五六日天])\s*(\d{1,2}):(\d{2})$", raw)
    if next_week_match:
        weekday_cn, hour_s, minute_s = next_week_match.groups()
        base = (reference_dt.astimezone(tz) if reference_dt else datetime.now(tz)).replace(
            second=0,
            microsecond=0,
        )
        target_wday = WEEKDAY_MAP[weekday_cn]
        current_wday = base.weekday()
        days_until = (target_wday - current_wday + 7) % 7
        days_until = days_until if days_until > 0 else 7
        target = base + timedelta(days=days_until)
        return target.replace(hour=int(hour_s), minute=int(minute_s))

    # Common formats
    now_local = reference_dt.astimezone(tz) if reference_dt else datetime.now(tz)
    patterns = [
        ("%Y-%m-%d %H:%M", True),
        ("%Y/%m/%d %H:%M", True),
        ("%Y-%m-%d %H:%M:%S", True),
        ("%Y/%m/%d %H:%M:%S", True),
        ("%Y年%m月%d日 %H:%M", True),
        ("%m-%d %H:%M", False),
        ("%m/%d %H:%M", False),
    ]
    for fmt, has_year in patterns:
        try:
            dt = datetime.strptime(raw, fmt)
            if not has_year:
                dt = dt.replace(year=now_local.year)
            return dt.replace(tzinfo=tz)
        except ValueError:
            continue

    raise ValueError(f"Unsupported datetime format: {value}")


def _format_local(utc_iso: str, timezone: str) -> str:
    tz_name = _normalize_timezone(timezone)
    dt_utc = datetime.fromisoformat(utc_iso)
    return dt_utc.astimezone(ZoneInfo(tz_name)).isoformat(timespec="minutes")


def _row_to_event(row: sqlite3.Row) -> Event:
    return Event(
        id=row["id"],
        title=row["title"],
        description=row["description"],
        start_time_utc=row["start_time_utc"],
        end_time_utc=row["end_time_utc"],
        timezone=row["timezone"],
        location=row["location"],
        source_type=row["source_type"],
        source_ref=row["source_ref"],
        source_line=row["source_line"],
        feishu_calendar_id=row["feishu_calendar_id"],
        feishu_event_id=row["feishu_event_id"],
        feishu_sync_error=row["feishu_sync_error"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _event_to_dict(event: Event, timezone: str | None = None) -> dict[str, Any]:
    tz_name = _normalize_timezone(timezone or event.timezone)
    return {
        "id": event.id,
        "title": event.title,
        "description": event.description,
        "start_time": _format_local(event.start_time_utc, tz_name),
        "end_time": _format_local(event.end_time_utc, tz_name),
        "timezone": tz_name,
        "location": event.location,
        "source_type": event.source_type,
        "source_ref": event.source_ref,
        "source_line": event.source_line,
        "feishu_calendar_id": event.feishu_calendar_id,
        "feishu_event_id": event.feishu_event_id,
        "feishu_sync_error": event.feishu_sync_error,
        "status": event.status,
        "created_at": event.created_at,
        "updated_at": event.updated_at,
    }


def _get_event_by_id(conn: sqlite3.Connection, event_id: str) -> Event | None:
    row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    if not row:
        return None
    return _row_to_event(row)


def _find_conflicts(
    conn: sqlite3.Connection,
    start_time_utc: str,
    end_time_utc: str,
    exclude_event_id: str | None = None,
) -> list[Event]:
    sql = (
        "SELECT * FROM events WHERE status = 'confirmed' "
        "AND NOT (end_time_utc <= ? OR start_time_utc >= ?)"
    )
    params: list[Any] = [start_time_utc, end_time_utc]
    if exclude_event_id:
        sql += " AND id != ?"
        params.append(exclude_event_id)
    sql += " ORDER BY start_time_utc ASC"
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_event(row) for row in rows]


def _insert_event(
    conn: sqlite3.Connection,
    *,
    title: str,
    description: str | None,
    start_dt: datetime,
    end_dt: datetime,
    timezone: str,
    location: str | None,
    source_type: str,
    source_ref: str | None,
    source_line: int | None,
) -> Event:
    now_iso = _now_utc_iso()
    event_id = str(uuid.uuid4())
    event = Event(
        id=event_id,
        title=title.strip(),
        description=(description or "").strip() or None,
        start_time_utc=_to_utc_iso(start_dt),
        end_time_utc=_to_utc_iso(end_dt),
        timezone=timezone,
        location=(location or "").strip() or None,
        source_type=source_type,
        source_ref=source_ref,
        source_line=source_line,
        feishu_calendar_id=None,
        feishu_event_id=None,
        feishu_sync_error=None,
        status="confirmed",
        created_at=now_iso,
        updated_at=now_iso,
    )
    conn.execute(
        """
        INSERT INTO events (
            id, title, description, start_time_utc, end_time_utc, timezone,
            location, source_type, source_ref, source_line,
            feishu_calendar_id, feishu_event_id, feishu_sync_error,
            status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.id,
            event.title,
            event.description,
            event.start_time_utc,
            event.end_time_utc,
            event.timezone,
            event.location,
            event.source_type,
            event.source_ref,
            event.source_line,
            event.feishu_calendar_id,
            event.feishu_event_id,
            event.feishu_sync_error,
            event.status,
            event.created_at,
            event.updated_at,
        ),
    )
    return event


def schedule_create(
    title: str,
    start_time: str,
    end_time: str | None = None,
    timezone: str = DEFAULT_TIMEZONE,
    description: str | None = None,
    location: str | None = None,
    source_type: str = "manual",
    source_ref: str | None = None,
    source_line: int | None = None,
    default_duration_minutes: int = DEFAULT_DURATION_MINUTES,
    allow_conflict: bool = False,
    sync_feishu_calendar: bool | None = None,
    feishu_calendar_id: str | None = None,
    app_id: str | None = None,
    app_secret: str | None = None,
    feishu_base_url: str = DEFAULT_FEISHU_BASE_URL,
    db_path: str | None = None,
) -> dict[str, str]:
    try:
        tz_name = _normalize_timezone(timezone)
        if not title or not title.strip():
            raise ValueError("title is required")

        start_dt = _parse_datetime(start_time, tz_name)
        if end_time:
            end_dt = _parse_datetime(end_time, tz_name)
        else:
            end_dt = start_dt + timedelta(minutes=max(default_duration_minutes, 1))

        if start_dt >= end_dt:
            raise ValueError("start_time must be earlier than end_time")

        init_db(db_path)
        with _connect(db_path) as conn:
            conflicts = _find_conflicts(conn, _to_utc_iso(start_dt), _to_utc_iso(end_dt))
            if conflicts and not allow_conflict:
                return _tool_response(
                    {
                        "ok": False,
                        "error": "time_conflict",
                        "message": "Found conflicting events. Retry with allow_conflict=true after user confirmation.",
                        "conflicts": [_event_to_dict(event, tz_name) for event in conflicts],
                    },
                    f"Create blocked by {len(conflicts)} conflict(s)",
                )

            created = _insert_event(
                conn,
                title=title,
                description=description,
                start_dt=start_dt,
                end_dt=end_dt,
                timezone=tz_name,
                location=location,
                source_type=source_type,
                source_ref=source_ref,
                source_line=source_line,
            )
            calendar_sync = None
            should_sync_calendar = (
                sync_feishu_calendar
                if sync_feishu_calendar is not None
                else _bool_env("FEISHU_SYNC_CALENDAR")
            )
            if should_sync_calendar:
                calendar_sync = _sync_event_to_feishu_calendar(
                    conn,
                    created,
                    action="create",
                    timezone=tz_name,
                    app_id=app_id,
                    app_secret=app_secret,
                    feishu_base_url=feishu_base_url,
                    calendar_id=feishu_calendar_id,
                )
            conn.commit()
            created = _get_event_by_id(conn, created.id) or created

        return _tool_response(
            {
                "ok": True,
                "event": _event_to_dict(created, tz_name),
                "conflict": bool(conflicts),
                "conflict_count": len(conflicts),
                "feishu_calendar_sync": calendar_sync,
            },
            f"Created event: {created.title}",
        )
    except Exception as exc:
        return _tool_response({"ok": False, "error": str(exc)}, "Failed to create event")


def schedule_query(
    keyword: str | None = None,
    start_from: str | None = None,
    end_to: str | None = None,
    timezone: str = DEFAULT_TIMEZONE,
    include_cancelled: bool = False,
    limit: int = 20,
    db_path: str | None = None,
) -> dict[str, str]:
    try:
        tz_name = _normalize_timezone(timezone)
        init_db(db_path)

        sql = "SELECT * FROM events WHERE 1=1"
        params: list[Any] = []

        if not include_cancelled:
            sql += " AND status != 'cancelled'"

        if keyword and keyword.strip():
            query = f"%{keyword.strip()}%"
            sql += " AND (title LIKE ? OR COALESCE(description, '') LIKE ? OR COALESCE(location, '') LIKE ?)"
            params.extend([query, query, query])

        if start_from:
            start_dt = _parse_datetime(start_from, tz_name)
            sql += " AND end_time_utc >= ?"
            params.append(_to_utc_iso(start_dt))

        if end_to:
            end_dt = _parse_datetime(end_to, tz_name)
            sql += " AND start_time_utc <= ?"
            params.append(_to_utc_iso(end_dt))

        safe_limit = max(1, min(int(limit), 200))
        sql += " ORDER BY start_time_utc ASC LIMIT ?"
        params.append(safe_limit)

        with _connect(db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
            events = [_event_to_dict(_row_to_event(row), tz_name) for row in rows]

        return _tool_response(
            {"ok": True, "count": len(events), "events": events},
            f"Found {len(events)} event(s)",
        )
    except Exception as exc:
        return _tool_response({"ok": False, "error": str(exc)}, "Failed to query events")


def schedule_update(
    event_id: str,
    title: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    timezone: str = DEFAULT_TIMEZONE,
    description: str | None = None,
    location: str | None = None,
    status: str | None = None,
    allow_conflict: bool = False,
    sync_feishu_calendar: bool | None = None,
    feishu_calendar_id: str | None = None,
    app_id: str | None = None,
    app_secret: str | None = None,
    feishu_base_url: str = DEFAULT_FEISHU_BASE_URL,
    db_path: str | None = None,
) -> dict[str, str]:
    try:
        tz_name = _normalize_timezone(timezone)
        init_db(db_path)
        with _connect(db_path) as conn:
            current = _get_event_by_id(conn, event_id)
            if current is None:
                return _tool_response({"ok": False, "error": f"event_id not found: {event_id}"}, "Event not found")

            if current.status == "cancelled":
                return _tool_response(
                    {"ok": False, "error": f"event_id is cancelled: {event_id}"},
                    "Cannot update cancelled event",
                )

            current_start = datetime.fromisoformat(current.start_time_utc).astimezone(ZoneInfo(tz_name))
            current_end = datetime.fromisoformat(current.end_time_utc).astimezone(ZoneInfo(tz_name))

            new_title = title.strip() if title is not None else current.title
            new_description = description if description is not None else current.description
            new_location = location if location is not None else current.location
            new_status = status.strip() if status is not None else current.status

            if new_status not in {"confirmed", "pending", "cancelled"}:
                raise ValueError("status must be one of: confirmed, pending, cancelled")

            new_start_dt = _parse_datetime(start_time, tz_name) if start_time else current_start
            new_end_dt = _parse_datetime(end_time, tz_name) if end_time else current_end

            if new_start_dt >= new_end_dt:
                raise ValueError("start_time must be earlier than end_time")

            conflicts = _find_conflicts(
                conn,
                _to_utc_iso(new_start_dt),
                _to_utc_iso(new_end_dt),
                exclude_event_id=event_id,
            )
            if conflicts and not allow_conflict:
                return _tool_response(
                    {
                        "ok": False,
                        "error": "time_conflict",
                        "message": "Found conflicting events. Retry with allow_conflict=true after user confirmation.",
                        "conflicts": [_event_to_dict(event, tz_name) for event in conflicts],
                    },
                    f"Update blocked by {len(conflicts)} conflict(s)",
                )

            now_iso = _now_utc_iso()
            conn.execute(
                """
                UPDATE events
                SET title = ?, description = ?, start_time_utc = ?, end_time_utc = ?,
                    timezone = ?, location = ?, status = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    new_title,
                    new_description,
                    _to_utc_iso(new_start_dt),
                    _to_utc_iso(new_end_dt),
                    tz_name,
                    new_location,
                    new_status,
                    now_iso,
                    event_id,
                ),
            )
            updated = _get_event_by_id(conn, event_id)
            assert updated is not None
            calendar_sync = None
            should_sync_calendar = (
                sync_feishu_calendar
                if sync_feishu_calendar is not None
                else _bool_env("FEISHU_SYNC_CALENDAR")
            )
            if should_sync_calendar:
                calendar_sync = _sync_event_to_feishu_calendar(
                    conn,
                    updated,
                    action="update",
                    timezone=tz_name,
                    app_id=app_id,
                    app_secret=app_secret,
                    feishu_base_url=feishu_base_url,
                    calendar_id=feishu_calendar_id,
                )
            conn.commit()

            updated = _get_event_by_id(conn, event_id)
            assert updated is not None

        return _tool_response(
            {
                "ok": True,
                "event": _event_to_dict(updated, tz_name),
                "conflict": bool(conflicts),
                "conflict_count": len(conflicts),
                "feishu_calendar_sync": calendar_sync,
            },
            f"Updated event: {updated.title}",
        )
    except Exception as exc:
        return _tool_response({"ok": False, "error": str(exc)}, "Failed to update event")


def schedule_delete(
    event_id: str,
    confirm: bool = False,
    sync_feishu_calendar: bool | None = None,
    feishu_calendar_id: str | None = None,
    app_id: str | None = None,
    app_secret: str | None = None,
    feishu_base_url: str = DEFAULT_FEISHU_BASE_URL,
    db_path: str | None = None,
) -> dict[str, str]:
    try:
        init_db(db_path)
        with _connect(db_path) as conn:
            current = _get_event_by_id(conn, event_id)
            if current is None:
                return _tool_response({"ok": False, "error": f"event_id not found: {event_id}"}, "Event not found")

            if not confirm:
                return _tool_response(
                    {
                        "ok": False,
                        "error": "confirmation_required",
                        "message": "Deletion needs confirm=true.",
                        "event": _event_to_dict(current, current.timezone),
                    },
                    "Deletion requires confirmation",
                )

            now_iso = _now_utc_iso()
            conn.execute(
                "UPDATE events SET status = 'cancelled', updated_at = ? WHERE id = ?",
                (now_iso, event_id),
            )
            deleted = _get_event_by_id(conn, event_id)
            assert deleted is not None
            calendar_sync = None
            should_sync_calendar = (
                sync_feishu_calendar
                if sync_feishu_calendar is not None
                else _bool_env("FEISHU_SYNC_CALENDAR")
            )
            if should_sync_calendar:
                calendar_sync = _sync_event_to_feishu_calendar(
                    conn,
                    deleted,
                    action="delete",
                    timezone=deleted.timezone,
                    app_id=app_id,
                    app_secret=app_secret,
                    feishu_base_url=feishu_base_url,
                    calendar_id=feishu_calendar_id,
                )
            conn.commit()

            deleted = _get_event_by_id(conn, event_id)
            assert deleted is not None

        return _tool_response(
            {
                "ok": True,
                "event": _event_to_dict(deleted, deleted.timezone),
                "feishu_calendar_sync": calendar_sync,
            },
            f"Cancelled event: {deleted.title}",
        )
    except Exception as exc:
        return _tool_response({"ok": False, "error": str(exc)}, "Failed to delete event")


def _remove_text_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text

    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if start >= end:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    chunks: list[str] = []
    cursor = 0
    for start, end in merged:
        chunks.append(text[cursor:start])
        cursor = end
    chunks.append(text[cursor:])
    return "".join(chunks)


def _clean_title(content: str, remove_spans: list[tuple[int, int]] | None = None) -> str:
    title = _remove_text_spans(content, remove_spans or [])
    title = title.replace("|", " ")
    title = LEADING_FILLER_PATTERN.sub("", title)
    title = re.sub(r"\s+", " ", title)
    title = title.strip(" -—:：\t")
    return title.strip()


def _strip_chat_prefix(line: str, timezone: str) -> tuple[str, datetime | None]:
    match = CHAT_PREFIX_PATTERN.match(line)
    if not match:
        return line, None

    timestamp = match.group("timestamp")
    body = match.group("body") or ""
    try:
        reference_dt = _parse_datetime(timestamp, timezone)
    except Exception:
        reference_dt = None
    return body.strip(), reference_dt


def _extract_event_from_line(
    line: str,
    line_no: int,
    timezone: str,
    default_duration_minutes: int,
    strip_chat_prefix: bool = False,
) -> dict[str, Any] | None:
    if not line.strip():
        return None

    content = line.strip()
    reference_dt = None
    if strip_chat_prefix:
        content, reference_dt = _strip_chat_prefix(line, timezone)
    content = BULLET_PREFIX_PATTERN.sub("", content).strip()

    dt_match = DATETIME_TOKEN_PATTERN.search(content)
    if not dt_match:
        return None

    start_token = dt_match.group(1)
    start_dt = _parse_datetime(start_token, timezone, reference_dt=reference_dt)

    range_match = TIME_RANGE_PATTERN.search(content)
    if range_match:
        end_hm = range_match.group(2)
        end_dt = start_dt.replace(
            hour=int(end_hm.split(":")[0]),
            minute=int(end_hm.split(":")[1]),
            second=0,
            microsecond=0,
        )
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)
    else:
        end_dt = start_dt + timedelta(minutes=max(default_duration_minutes, 1))

    remove_spans = [dt_match.span()]
    if range_match:
        remove_spans.append(range_match.span())

    title = _clean_title(content, remove_spans=remove_spans)
    if not title:
        title = f"未命名日程(line {line_no})"

    return {
        "title": title,
        "description": content.strip(),
        "start_dt": start_dt,
        "end_dt": end_dt,
        "line_no": line_no,
    }


def _find_same_day_event_by_title(
    conn: sqlite3.Connection,
    title: str,
    start_dt: datetime,
    timezone: str,
) -> Event | None:
    tz = ZoneInfo(_normalize_timezone(timezone))
    local_day_start = start_dt.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    local_day_end = local_day_start + timedelta(days=1)

    day_start_utc = _to_utc_iso(local_day_start)
    day_end_utc = _to_utc_iso(local_day_end)

    row = conn.execute(
        """
        SELECT * FROM events
        WHERE status != 'cancelled'
          AND title = ?
          AND start_time_utc >= ?
          AND start_time_utc < ?
        ORDER BY start_time_utc ASC
        LIMIT 1
        """,
        (title.strip(), day_start_utc, day_end_utc),
    ).fetchone()
    if not row:
        return None
    return _row_to_event(row)


def _ingest_text(
    text: str,
    *,
    source_type: str,
    source_ref: str,
    timezone: str,
    mode: str,
    default_duration_minutes: int,
    allow_conflict: bool,
    sync_feishu_calendar: bool | None,
    feishu_calendar_id: str | None,
    app_id: str | None,
    app_secret: str | None,
    feishu_base_url: str,
    strip_chat_prefix: bool,
    db_path: str | None,
) -> dict[str, str]:
    tz_name = _normalize_timezone(timezone)
    if mode not in {"create", "upsert"}:
        return _tool_response({"ok": False, "error": "mode must be create or upsert"}, "Invalid ingest mode")

    init_db(db_path)
    created_ids: list[str] = []
    updated_ids: list[str] = []
    skipped_lines: list[int] = []
    conflicts_total = 0
    synced_created = 0
    synced_updated = 0
    sync_errors: list[dict[str, Any]] = []

    with _connect(db_path) as conn:
        lines = text.splitlines()
        for idx, line in enumerate(lines, start=1):
            candidate = _extract_event_from_line(
                line,
                idx,
                tz_name,
                default_duration_minutes,
                strip_chat_prefix=strip_chat_prefix,
            )
            if not candidate:
                continue

            title = candidate["title"]
            description = candidate["description"]
            start_dt = candidate["start_dt"]
            end_dt = candidate["end_dt"]
            source_line = candidate["line_no"]

            if mode == "upsert":
                existing = _find_same_day_event_by_title(conn, title, start_dt, tz_name)
                if existing is not None:
                    conflicts = _find_conflicts(
                        conn,
                        _to_utc_iso(start_dt),
                        _to_utc_iso(end_dt),
                        exclude_event_id=existing.id,
                    )
                    if conflicts and not allow_conflict:
                        skipped_lines.append(source_line)
                        conflicts_total += len(conflicts)
                        continue

                    now_iso = _now_utc_iso()
                    conn.execute(
                        """
                        UPDATE events
                        SET description = ?, start_time_utc = ?, end_time_utc = ?,
                            timezone = ?, source_type = ?, source_ref = ?, source_line = ?,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            description,
                            _to_utc_iso(start_dt),
                            _to_utc_iso(end_dt),
                            tz_name,
                            source_type,
                            source_ref,
                            source_line,
                            now_iso,
                            existing.id,
                        ),
                    )
                    updated_event = _get_event_by_id(conn, existing.id)
                    assert updated_event is not None
                    should_sync_calendar = (
                        sync_feishu_calendar
                        if sync_feishu_calendar is not None
                        else _bool_env("FEISHU_SYNC_CALENDAR")
                    )
                    if should_sync_calendar:
                        sync_result = _sync_event_to_feishu_calendar(
                            conn,
                            updated_event,
                            action="update",
                            timezone=tz_name,
                            app_id=app_id,
                            app_secret=app_secret,
                            feishu_base_url=feishu_base_url,
                            calendar_id=feishu_calendar_id,
                        )
                        if sync_result.get("ok"):
                            synced_updated += 1
                        else:
                            sync_errors.append(
                                {
                                    "line": source_line,
                                    "title": title,
                                    "error": sync_result.get("error"),
                                }
                            )
                    updated_ids.append(existing.id)
                    continue

            conflicts = _find_conflicts(conn, _to_utc_iso(start_dt), _to_utc_iso(end_dt))
            if conflicts and not allow_conflict:
                skipped_lines.append(source_line)
                conflicts_total += len(conflicts)
                continue

            event = _insert_event(
                conn,
                title=title,
                description=description,
                start_dt=start_dt,
                end_dt=end_dt,
                timezone=tz_name,
                location=None,
                source_type=source_type,
                source_ref=source_ref,
                source_line=source_line,
            )
            should_sync_calendar = (
                sync_feishu_calendar
                if sync_feishu_calendar is not None
                else _bool_env("FEISHU_SYNC_CALENDAR")
            )
            if should_sync_calendar:
                sync_result = _sync_event_to_feishu_calendar(
                    conn,
                    event,
                    action="create",
                    timezone=tz_name,
                    app_id=app_id,
                    app_secret=app_secret,
                    feishu_base_url=feishu_base_url,
                    calendar_id=feishu_calendar_id,
                )
                if sync_result.get("ok"):
                    synced_created += 1
                else:
                    sync_errors.append(
                        {
                            "line": source_line,
                            "title": title,
                            "error": sync_result.get("error"),
                        }
                    )
            created_ids.append(event.id)

        conn.commit()

    return _tool_response(
        {
            "ok": True,
            "mode": mode,
            "source_type": source_type,
            "source_ref": source_ref,
            "created_count": len(created_ids),
            "updated_count": len(updated_ids),
            "skipped_count": len(skipped_lines),
            "skipped_lines": skipped_lines,
            "conflicts_total": conflicts_total,
            "created_ids": created_ids,
            "updated_ids": updated_ids,
            "feishu_calendar_sync": {
                "enabled": (
                    sync_feishu_calendar
                    if sync_feishu_calendar is not None
                    else _bool_env("FEISHU_SYNC_CALENDAR")
                ),
                "synced_created_count": synced_created,
                "synced_updated_count": synced_updated,
                "errors": sync_errors,
            },
        },
        f"Ingest done: +{len(created_ids)} / ~{len(updated_ids)} / skip {len(skipped_lines)}",
    )


def _http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    req_headers = {"Content-Type": "application/json; charset=utf-8"}
    if headers:
        req_headers.update(headers)
    req = Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=req_headers,
        method="POST",
    )
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Feishu HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Feishu request failed: {exc.reason}") from exc


def _http_patch_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    req_headers = {"Content-Type": "application/json; charset=utf-8"}
    if headers:
        req_headers.update(headers)
    req = Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=req_headers,
        method="PATCH",
    )
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Feishu HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Feishu request failed: {exc.reason}") from exc


def _http_delete_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    req = Request(url=url, headers=headers or {}, method="DELETE")
    try:
        with urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body) if body else {"code": 0}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Feishu HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Feishu request failed: {exc.reason}") from exc


def _http_get_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    req = Request(url=url, headers=headers or {}, method="GET")
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Feishu HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Feishu request failed: {exc.reason}") from exc


def _extract_feishu_resource(document_id_or_url: str) -> tuple[str, str]:
    raw = (document_id_or_url or "").strip()
    if not raw:
        raise ValueError("document_id_or_url is required")

    if "://" not in raw:
        candidate = raw.strip("/")
        if re.fullmatch(r"[A-Za-z0-9_-]+", candidate):
            return ("docx", candidate)
        raise ValueError(f"Invalid document id: {raw}")

    parsed = urlparse(raw)
    path = parsed.path or ""
    doc_patterns = [
        r"/docx/([A-Za-z0-9_-]+)",
        r"/docs/([A-Za-z0-9_-]+)",
    ]
    for pattern in doc_patterns:
        match = re.search(pattern, path)
        if match:
            return ("docx", match.group(1))

    wiki_match = re.search(r"/wiki/([A-Za-z0-9_-]+)", path)
    if wiki_match:
        return ("wiki", wiki_match.group(1))

    raise ValueError(f"Could not parse document id or wiki token from url: {raw}")


def _get_feishu_tenant_access_token(app_id: str, app_secret: str, feishu_base_url: str) -> str:
    endpoint = feishu_base_url.rstrip("/") + "/open-apis/auth/v3/tenant_access_token/internal"
    data = _http_post_json(endpoint, {"app_id": app_id, "app_secret": app_secret})
    if int(data.get("code", -1)) != 0:
        raise RuntimeError(f"Feishu auth failed: code={data.get('code')} msg={data.get('msg')}")
    token = data.get("tenant_access_token")
    if not token:
        raise RuntimeError("Feishu auth succeeded but tenant_access_token is missing")
    return str(token)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_feishu_credentials(
    app_id: str | None = None,
    app_secret: str | None = None,
) -> tuple[str, str]:
    resolved_app_id = (app_id or os.getenv("FEISHU_APP_ID") or "").strip()
    resolved_app_secret = (app_secret or os.getenv("FEISHU_APP_SECRET") or "").strip()
    if not resolved_app_id or not resolved_app_secret:
        raise RuntimeError("Provide app_id/app_secret or set FEISHU_APP_ID and FEISHU_APP_SECRET.")
    return resolved_app_id, resolved_app_secret


def _feishu_auth_header(tenant_access_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {tenant_access_token}"}


def _feishu_calendar_id_from_env() -> str:
    return (os.getenv("FEISHU_CALENDAR_ID") or "primary").strip() or "primary"


def _feishu_event_time(dt: datetime, timezone: str) -> dict[str, str]:
    return {"timestamp": str(int(dt.astimezone(UTC).timestamp()))}


def _feishu_calendar_event_payload(event: Event, timezone: str) -> dict[str, Any]:
    start_dt = datetime.fromisoformat(event.start_time_utc).astimezone(ZoneInfo(timezone))
    end_dt = datetime.fromisoformat(event.end_time_utc).astimezone(ZoneInfo(timezone))
    return {
        "summary": event.title,
        "description": event.description or "",
        "start_time": _feishu_event_time(start_dt, timezone),
        "end_time": _feishu_event_time(end_dt, timezone),
    }


def _feishu_calendar_create_event(
    event: Event,
    *,
    calendar_id: str,
    tenant_access_token: str,
    feishu_base_url: str,
    timezone: str,
) -> tuple[str, dict[str, Any]]:
    endpoint = (
        feishu_base_url.rstrip("/")
        + f"/open-apis/calendar/v4/calendars/{quote(calendar_id, safe='')}/events"
    )
    data = _http_post_json(
        endpoint,
        _feishu_calendar_event_payload(event, timezone),
        headers=_feishu_auth_header(tenant_access_token),
    )
    if int(data.get("code", -1)) != 0:
        raise RuntimeError(f"Feishu calendar create failed: code={data.get('code')} msg={data.get('msg')}")
    event_obj = (data.get("data") or {}).get("event") or {}
    feishu_event_id = event_obj.get("event_id") or event_obj.get("id")
    if not feishu_event_id:
        raise RuntimeError("Feishu calendar create succeeded but event_id is missing")
    return str(feishu_event_id), data


def _feishu_calendar_update_event(
    event: Event,
    *,
    calendar_id: str,
    feishu_event_id: str,
    tenant_access_token: str,
    feishu_base_url: str,
    timezone: str,
) -> dict[str, Any]:
    endpoint = (
        feishu_base_url.rstrip("/")
        + f"/open-apis/calendar/v4/calendars/{quote(calendar_id, safe='')}/events/{quote(feishu_event_id, safe='')}"
    )
    data = _http_patch_json(
        endpoint,
        _feishu_calendar_event_payload(event, timezone),
        headers=_feishu_auth_header(tenant_access_token),
    )
    if int(data.get("code", -1)) != 0:
        raise RuntimeError(f"Feishu calendar update failed: code={data.get('code')} msg={data.get('msg')}")
    return data


def _feishu_calendar_delete_event(
    *,
    calendar_id: str,
    feishu_event_id: str,
    tenant_access_token: str,
    feishu_base_url: str,
) -> dict[str, Any]:
    endpoint = (
        feishu_base_url.rstrip("/")
        + f"/open-apis/calendar/v4/calendars/{quote(calendar_id, safe='')}/events/{quote(feishu_event_id, safe='')}"
    )
    data = _http_delete_json(endpoint, headers=_feishu_auth_header(tenant_access_token))
    if int(data.get("code", -1)) != 0:
        raise RuntimeError(f"Feishu calendar delete failed: code={data.get('code')} msg={data.get('msg')}")
    return data


def _sync_event_to_feishu_calendar(
    conn: sqlite3.Connection,
    event: Event,
    *,
    action: str,
    timezone: str,
    app_id: str | None = None,
    app_secret: str | None = None,
    feishu_base_url: str = DEFAULT_FEISHU_BASE_URL,
    calendar_id: str | None = None,
) -> dict[str, Any]:
    target_calendar_id = (calendar_id or _feishu_calendar_id_from_env()).strip()
    resolved_app_id, resolved_app_secret = _get_feishu_credentials(app_id, app_secret)
    token = _get_feishu_tenant_access_token(resolved_app_id, resolved_app_secret, feishu_base_url)

    try:
        if action == "create":
            feishu_event_id, raw = _feishu_calendar_create_event(
                event,
                calendar_id=target_calendar_id,
                tenant_access_token=token,
                feishu_base_url=feishu_base_url,
                timezone=timezone,
            )
            conn.execute(
                """
                UPDATE events
                SET feishu_calendar_id = ?, feishu_event_id = ?, feishu_sync_error = NULL
                WHERE id = ?
                """,
                (target_calendar_id, feishu_event_id, event.id),
            )
            return {"ok": True, "action": action, "calendar_id": target_calendar_id, "event_id": feishu_event_id, "raw": raw}

        if action == "update":
            if event.feishu_event_id:
                raw = _feishu_calendar_update_event(
                    event,
                    calendar_id=event.feishu_calendar_id or target_calendar_id,
                    feishu_event_id=event.feishu_event_id,
                    tenant_access_token=token,
                    feishu_base_url=feishu_base_url,
                    timezone=timezone,
                )
                conn.execute(
                    "UPDATE events SET feishu_sync_error = NULL WHERE id = ?",
                    (event.id,),
                )
                return {"ok": True, "action": action, "calendar_id": event.feishu_calendar_id or target_calendar_id, "event_id": event.feishu_event_id, "raw": raw}

            return _sync_event_to_feishu_calendar(
                conn,
                event,
                action="create",
                timezone=timezone,
                app_id=app_id,
                app_secret=app_secret,
                feishu_base_url=feishu_base_url,
                calendar_id=target_calendar_id,
            )

        if action == "delete":
            if not event.feishu_event_id:
                return {"ok": True, "action": action, "skipped": True, "reason": "no_feishu_event_id"}
            raw = _feishu_calendar_delete_event(
                calendar_id=event.feishu_calendar_id or target_calendar_id,
                feishu_event_id=event.feishu_event_id,
                tenant_access_token=token,
                feishu_base_url=feishu_base_url,
            )
            conn.execute(
                """
                UPDATE events
                SET feishu_event_id = NULL, feishu_sync_error = NULL
                WHERE id = ?
                """,
                (event.id,),
            )
            return {"ok": True, "action": action, "calendar_id": event.feishu_calendar_id or target_calendar_id, "event_id": event.feishu_event_id, "raw": raw}

        raise ValueError(f"Unsupported Feishu calendar sync action: {action}")
    except Exception as exc:
        message = str(exc)
        conn.execute(
            "UPDATE events SET feishu_sync_error = ? WHERE id = ?",
            (message, event.id),
        )
        return {"ok": False, "action": action, "calendar_id": target_calendar_id, "error": message}


def _get_feishu_doc_raw_content(document_id: str, tenant_access_token: str, feishu_base_url: str) -> str:
    endpoint = feishu_base_url.rstrip("/") + f"/open-apis/docx/v1/documents/{document_id}/raw_content"
    data = _http_get_json(endpoint, headers={"Authorization": f"Bearer {tenant_access_token}"})
    if int(data.get("code", -1)) != 0:
        raise RuntimeError(f"Feishu doc read failed: code={data.get('code')} msg={data.get('msg')}")

    content = (data.get("data") or {}).get("content")
    if not isinstance(content, str):
        raise RuntimeError("Feishu doc read succeeded but data.content is missing")
    return content


def _resolve_doc_id_from_wiki_node(
    wiki_node_token: str,
    tenant_access_token: str,
    feishu_base_url: str,
) -> str:
    endpoint = (
        feishu_base_url.rstrip("/")
        + "/open-apis/wiki/v2/spaces/get_node?token="
        + quote(wiki_node_token, safe="")
    )
    data = _http_get_json(endpoint, headers={"Authorization": f"Bearer {tenant_access_token}"})
    if int(data.get("code", -1)) != 0:
        raise RuntimeError(f"Feishu wiki read failed: code={data.get('code')} msg={data.get('msg')}")

    node = ((data.get("data") or {}).get("node") or {})
    obj_type = node.get("obj_type")
    obj_token = node.get("obj_token")
    if obj_type != "docx" or not isinstance(obj_token, str) or not obj_token.strip():
        raise RuntimeError(f"Wiki node is not a docx document (obj_type={obj_type})")
    return obj_token.strip()


def ingest_markdown_schedules(
    file_path: str,
    timezone: str = DEFAULT_TIMEZONE,
    mode: str = "upsert",
    default_duration_minutes: int = DEFAULT_DURATION_MINUTES,
    allow_conflict: bool = True,
    sync_feishu_calendar: bool | None = None,
    feishu_calendar_id: str | None = None,
    app_id: str | None = None,
    app_secret: str | None = None,
    feishu_base_url: str = DEFAULT_FEISHU_BASE_URL,
    db_path: str | None = None,
) -> dict[str, str]:
    try:
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            return _tool_response({"ok": False, "error": f"file not found: {path}"}, "Markdown file not found")
        text = path.read_text(encoding="utf-8", errors="replace")
        return _ingest_text(
            text,
            source_type="markdown",
            source_ref=str(path),
            timezone=timezone,
            mode=mode,
            default_duration_minutes=default_duration_minutes,
            allow_conflict=allow_conflict,
            sync_feishu_calendar=sync_feishu_calendar,
            feishu_calendar_id=feishu_calendar_id,
            app_id=app_id,
            app_secret=app_secret,
            feishu_base_url=feishu_base_url,
            strip_chat_prefix=False,
            db_path=db_path,
        )
    except Exception as exc:
        return _tool_response({"ok": False, "error": str(exc)}, "Failed to ingest markdown")


def ingest_chat_schedules(
    file_path: str,
    timezone: str = DEFAULT_TIMEZONE,
    mode: str = "upsert",
    default_duration_minutes: int = DEFAULT_DURATION_MINUTES,
    allow_conflict: bool = True,
    sync_feishu_calendar: bool | None = None,
    feishu_calendar_id: str | None = None,
    app_id: str | None = None,
    app_secret: str | None = None,
    feishu_base_url: str = DEFAULT_FEISHU_BASE_URL,
    db_path: str | None = None,
) -> dict[str, str]:
    try:
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            return _tool_response({"ok": False, "error": f"file not found: {path}"}, "Chat file not found")
        text = path.read_text(encoding="utf-8", errors="replace")
        return _ingest_text(
            text,
            source_type="chat",
            source_ref=str(path),
            timezone=timezone,
            mode=mode,
            default_duration_minutes=default_duration_minutes,
            allow_conflict=allow_conflict,
            sync_feishu_calendar=sync_feishu_calendar,
            feishu_calendar_id=feishu_calendar_id,
            app_id=app_id,
            app_secret=app_secret,
            feishu_base_url=feishu_base_url,
            strip_chat_prefix=True,
            db_path=db_path,
        )
    except Exception as exc:
        return _tool_response({"ok": False, "error": str(exc)}, "Failed to ingest chat")


def ingest_feishu_markdown_schedules(
    file_path: str,
    timezone: str = DEFAULT_TIMEZONE,
    mode: str = "upsert",
    default_duration_minutes: int = DEFAULT_DURATION_MINUTES,
    allow_conflict: bool = True,
    sync_feishu_calendar: bool | None = None,
    feishu_calendar_id: str | None = None,
    app_id: str | None = None,
    app_secret: str | None = None,
    feishu_base_url: str = DEFAULT_FEISHU_BASE_URL,
    db_path: str | None = None,
) -> dict[str, str]:
    """Ingest Feishu docs exported as markdown.

    This keeps the MVP implementation simple and practical: users can export
    Feishu docs as markdown and run the same ingestion pipeline.
    """

    try:
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            return _tool_response({"ok": False, "error": f"file not found: {path}"}, "Feishu markdown not found")
        text = path.read_text(encoding="utf-8", errors="replace")
        return _ingest_text(
            text,
            source_type="feishu_markdown",
            source_ref=str(path),
            timezone=timezone,
            mode=mode,
            default_duration_minutes=default_duration_minutes,
            allow_conflict=allow_conflict,
            sync_feishu_calendar=sync_feishu_calendar,
            feishu_calendar_id=feishu_calendar_id,
            app_id=app_id,
            app_secret=app_secret,
            feishu_base_url=feishu_base_url,
            strip_chat_prefix=False,
            db_path=db_path,
        )
    except Exception as exc:
        return _tool_response({"ok": False, "error": str(exc)}, "Failed to ingest feishu markdown")


def ingest_feishu_doc_schedules(
    document_id_or_url: str,
    timezone: str = DEFAULT_TIMEZONE,
    mode: str = "upsert",
    default_duration_minutes: int = DEFAULT_DURATION_MINUTES,
    allow_conflict: bool = True,
    sync_feishu_calendar: bool | None = None,
    feishu_calendar_id: str | None = None,
    app_id: str | None = None,
    app_secret: str | None = None,
    feishu_base_url: str = DEFAULT_FEISHU_BASE_URL,
    db_path: str | None = None,
) -> dict[str, str]:
    """Ingest schedules from Feishu Docx API directly.

    Credentials are read from args first, then env vars:
    - FEISHU_APP_ID
    - FEISHU_APP_SECRET
    """

    try:
        try:
            resolved_app_id, resolved_app_secret = _get_feishu_credentials(app_id, app_secret)
        except RuntimeError:
            return _tool_response(
                {
                    "ok": False,
                    "error": "missing_feishu_credentials",
                    "message": "Provide app_id/app_secret or set FEISHU_APP_ID and FEISHU_APP_SECRET.",
                },
                "Missing Feishu credentials",
            )

        ref_type, ref_token = _extract_feishu_resource(document_id_or_url)
        token = _get_feishu_tenant_access_token(
            app_id=resolved_app_id,
            app_secret=resolved_app_secret,
            feishu_base_url=feishu_base_url,
        )
        document_id = (
            _resolve_doc_id_from_wiki_node(ref_token, token, feishu_base_url)
            if ref_type == "wiki"
            else ref_token
        )
        raw_content = _get_feishu_doc_raw_content(
            document_id=document_id,
            tenant_access_token=token,
            feishu_base_url=feishu_base_url,
        )

        return _ingest_text(
            raw_content,
            source_type="feishu_doc",
            source_ref=f"feishu://docx/{document_id}",
            timezone=timezone,
            mode=mode,
            default_duration_minutes=default_duration_minutes,
            allow_conflict=allow_conflict,
            sync_feishu_calendar=sync_feishu_calendar,
            feishu_calendar_id=feishu_calendar_id,
            app_id=resolved_app_id,
            app_secret=resolved_app_secret,
            feishu_base_url=feishu_base_url,
            strip_chat_prefix=False,
            db_path=db_path,
        )
    except Exception as exc:
        return _tool_response({"ok": False, "error": str(exc)}, "Failed to ingest feishu doc")
