from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from exercise1_nexau import scheduler_tools as st

BASE_DIR = Path(__file__).parent.resolve()
TEMPLATE_DIR = BASE_DIR / "templates"
DEFAULT_DEMO_DB_PATH = BASE_DIR / "data" / "demo_schedule.db"
DEFAULT_SAMPLE_NOTES_PATH = BASE_DIR / "data" / "sample_notes.md"


class CreateEventRequest(BaseModel):
    title: str
    start_time: str
    end_time: str | None = None
    timezone: str = st.DEFAULT_TIMEZONE
    description: str | None = None
    location: str | None = None


class UpdateEventRequest(BaseModel):
    title: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    timezone: str = st.DEFAULT_TIMEZONE
    description: str | None = None
    location: str | None = None


class FeishuImportRequest(BaseModel):
    document_url: str
    mode: str = "upsert"
    timezone: str = st.DEFAULT_TIMEZONE
    sync_feishu_calendar: bool = True
    feishu_calendar_id: str | None = None


def _decode_tool_response(response: dict[str, str]) -> dict[str, Any]:
    payload = json.loads(response["content"])
    payload["return_display"] = response.get("returnDisplay")
    return payload


def _clear_events(db_path: str) -> None:
    st.init_db(db_path=db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM events")
        conn.commit()


def create_app(
    *,
    db_path: str | None = None,
    timezone: str = st.DEFAULT_TIMEZONE,
    sample_notes_path: str | None = None,
) -> FastAPI:
    resolved_db_path = str(Path(db_path or DEFAULT_DEMO_DB_PATH).expanduser().resolve())
    resolved_sample_notes = str(
        Path(sample_notes_path or DEFAULT_SAMPLE_NOTES_PATH).expanduser().resolve()
    )
    st.init_db(db_path=resolved_db_path)

    app = FastAPI(title="Exercise1 Schedule Demo")
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="schedule_demo.html",
            context={
                "timezone": timezone,
                "db_path": resolved_db_path,
                "sample_notes_path": resolved_sample_notes,
            },
        )

    @app.get("/api/events")
    async def list_events(
        keyword: str = Query(default=""),
        include_cancelled: bool = Query(default=True),
    ) -> dict[str, Any]:
        response = st.schedule_query(
            keyword=keyword or None,
            timezone=timezone,
            include_cancelled=include_cancelled,
            limit=100,
            db_path=resolved_db_path,
        )
        payload = _decode_tool_response(response)
        if not payload.get("ok"):
            raise HTTPException(status_code=400, detail=payload)
        return payload

    @app.post("/api/events")
    async def create_event(body: CreateEventRequest) -> dict[str, Any]:
        response = st.schedule_create(
            title=body.title,
            start_time=body.start_time,
            end_time=body.end_time,
            timezone=body.timezone,
            description=body.description,
            location=body.location,
            db_path=resolved_db_path,
        )
        payload = _decode_tool_response(response)
        if not payload.get("ok"):
            raise HTTPException(status_code=400, detail=payload)
        return payload

    @app.patch("/api/events/{event_id}")
    async def update_event(event_id: str, body: UpdateEventRequest) -> dict[str, Any]:
        response = st.schedule_update(
            event_id=event_id,
            title=body.title,
            start_time=body.start_time,
            end_time=body.end_time,
            timezone=body.timezone,
            description=body.description,
            location=body.location,
            db_path=resolved_db_path,
        )
        payload = _decode_tool_response(response)
        if not payload.get("ok"):
            raise HTTPException(status_code=400, detail=payload)
        return payload

    @app.post("/api/events/{event_id}/cancel")
    async def cancel_event(event_id: str) -> dict[str, Any]:
        response = st.schedule_delete(
            event_id=event_id,
            confirm=True,
            db_path=resolved_db_path,
        )
        payload = _decode_tool_response(response)
        if not payload.get("ok"):
            raise HTTPException(status_code=400, detail=payload)
        return payload

    @app.post("/api/demo/import-sample")
    async def import_sample() -> dict[str, Any]:
        sample_path = Path(resolved_sample_notes)
        if not sample_path.exists():
            raise HTTPException(
                status_code=404,
                detail={"ok": False, "error": f"sample notes not found: {sample_path}"},
            )

        response = st.ingest_markdown_schedules(
            file_path=str(sample_path),
            timezone=timezone,
            mode="upsert",
            db_path=resolved_db_path,
        )
        payload = _decode_tool_response(response)
        if not payload.get("ok"):
            raise HTTPException(status_code=400, detail=payload)
        return payload

    @app.post("/api/demo/import-feishu")
    async def import_feishu(body: FeishuImportRequest) -> dict[str, Any]:
        response = st.ingest_feishu_doc_schedules(
            document_id_or_url=body.document_url,
            timezone=body.timezone,
            mode=body.mode,
            sync_feishu_calendar=body.sync_feishu_calendar,
            feishu_calendar_id=body.feishu_calendar_id,
            db_path=resolved_db_path,
        )
        payload = _decode_tool_response(response)
        if not payload.get("ok"):
            raise HTTPException(status_code=400, detail=payload)
        return payload

    @app.post("/api/demo/reset")
    async def reset_demo() -> dict[str, Any]:
        _clear_events(resolved_db_path)
        return {
            "ok": True,
            "message": "Demo database reset",
            "db_path": resolved_db_path,
        }

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exercise1 schedule demo web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--db-path", default=str(DEFAULT_DEMO_DB_PATH))
    parser.add_argument("--timezone", default=st.DEFAULT_TIMEZONE)
    parser.add_argument("--sample-notes-path", default=str(DEFAULT_SAMPLE_NOTES_PATH))
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        create_app(
            db_path=args.db_path,
            timezone=args.timezone,
            sample_notes_path=args.sample_notes_path,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
