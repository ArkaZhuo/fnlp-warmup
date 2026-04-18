"""Run exercise-1 personal schedule assistant with NexAU.

Usage:
  python run_scheduler_agent.py --message "明天下午3点安排项目复盘"
  python run_scheduler_agent.py
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from nexau import Agent, AgentConfig, LLMConfig, Tool
from nexau.archs.main_sub.execution.hooks import (
    AfterModelHookInput,
    AfterModelHookResult,
    AfterToolHookInput,
    AfterToolHookResult,
    HookResult,
)

try:
    from .scheduler_tools import (
        DEFAULT_DB_PATH,
        DEFAULT_FEISHU_BASE_URL,
        DEFAULT_TIMEZONE,
        ingest_chat_schedules,
        ingest_feishu_doc_schedules,
        ingest_feishu_markdown_schedules,
        ingest_markdown_schedules,
        init_db,
        schedule_create,
        schedule_delete,
        schedule_query,
        schedule_update,
    )
except ImportError:  # pragma: no cover - keeps direct script execution working
    from scheduler_tools import (
        DEFAULT_DB_PATH,
        DEFAULT_FEISHU_BASE_URL,
        DEFAULT_TIMEZONE,
        ingest_chat_schedules,
        ingest_feishu_doc_schedules,
        ingest_feishu_markdown_schedules,
        ingest_markdown_schedules,
        init_db,
        schedule_create,
        schedule_delete,
        schedule_query,
        schedule_update,
    )

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def _load_local_env(base_dir: Path) -> None:
    if load_dotenv is None:
        return

    candidates = [
        base_dir.parent / ".env",
        base_dir / ".env",
    ]
    loaded_any = False
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate, override=False)
            loaded_any = True

    if not loaded_any:
        load_dotenv()


def _now_string(timezone: str) -> str:
    tz = ZoneInfo(timezone)
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")


def _format_preview_value(value: Any, limit: int = 80) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _format_tool_input_preview(tool_input: dict[str, Any], limit: int = 3) -> str:
    if not tool_input:
        return ""

    parts: list[str] = []
    for index, (key, value) in enumerate(tool_input.items()):
        if index >= limit:
            parts.append("...")
            break
        parts.append(f"{key}={_format_preview_value(value)}")
    return ", ".join(parts)


def _extract_tool_payload(tool_output: Any) -> dict[str, Any] | None:
    if not isinstance(tool_output, dict):
        return None

    content = tool_output.get("content")
    if not isinstance(content, str):
        return None

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _summarize_tool_payload(tool_name: str, payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None

    if payload.get("ok") is False:
        return payload.get("error") or payload.get("message") or "执行失败"

    event = payload.get("event")
    if isinstance(event, dict):
        title = event.get("title")
        event_id = event.get("id")
        parts = []
        if title:
            parts.append(f"title={title}")
        if event_id:
            parts.append(f"event_id={event_id}")
        return ", ".join(parts) if parts else None

    if tool_name == "schedule_query":
        return f"count={payload.get('count', 0)}"

    if any(key in payload for key in ("created_count", "updated_count", "skipped_count")):
        return (
            f"新增={payload.get('created_count', 0)}, "
            f"更新={payload.get('updated_count', 0)}, "
            f"跳过={payload.get('skipped_count', 0)}"
        )

    if "db_path" in payload:
        return f"db_path={payload['db_path']}"

    return None


def _build_trace_hooks(
    trace_callback: Callable[[str], None],
) -> tuple[list[Callable[..., Any]], list[Callable[..., Any]], list[Callable[..., Any]]]:
    def emit(message: str) -> None:
        try:
            trace_callback(message)
        except Exception:
            pass

    def after_model_hook(hook_input: AfterModelHookInput) -> AfterModelHookResult:
        parsed = hook_input.parsed_response
        if parsed and parsed.tool_calls:
            names: list[str] = []
            for call in parsed.tool_calls:
                preview = _format_tool_input_preview(getattr(call, "tool_input", {}) or {})
                if preview:
                    names.append(f"{call.tool_name}({preview})")
                else:
                    names.append(call.tool_name)
            emit("Agent 计划调用工具: " + " -> ".join(names))
        return AfterModelHookResult.no_changes()

    def before_tool_hook(hook_input) -> HookResult:
        preview = _format_tool_input_preview(hook_input.tool_input)
        if preview:
            emit(f"正在调用工具 {hook_input.tool_name}({preview})")
        else:
            emit(f"正在调用工具 {hook_input.tool_name}")
        return HookResult.no_changes()

    def after_tool_hook(hook_input: AfterToolHookInput) -> AfterToolHookResult:
        payload = _extract_tool_payload(hook_input.tool_output)
        status = "完成"
        if isinstance(payload, dict) and payload.get("ok") is False:
            status = "失败"
        summary = _summarize_tool_payload(hook_input.tool_name, payload)
        if summary:
            emit(f"工具 {hook_input.tool_name} {status} | {summary}")
        else:
            emit(f"工具 {hook_input.tool_name} {status}")
        return AfterToolHookResult.no_changes()

    return [after_model_hook], [before_tool_hook], [after_tool_hook]


def _build_agent(
    base_dir: Path,
    timezone: str,
    db_path: str,
    trace_callback: Callable[[str], None] | None = None,
) -> Agent:
    tool_dir = base_dir / "tools"

    tools = [
        Tool.from_yaml(
            str(tool_dir / "schedule_create.tool.yaml"),
            binding=schedule_create,
            extra_kwargs={"db_path": db_path, "timezone": timezone},
        ),
        Tool.from_yaml(
            str(tool_dir / "schedule_query.tool.yaml"),
            binding=schedule_query,
            extra_kwargs={"db_path": db_path, "timezone": timezone},
        ),
        Tool.from_yaml(
            str(tool_dir / "schedule_update.tool.yaml"),
            binding=schedule_update,
            extra_kwargs={"db_path": db_path, "timezone": timezone},
        ),
        Tool.from_yaml(
            str(tool_dir / "schedule_delete.tool.yaml"),
            binding=schedule_delete,
            extra_kwargs={"db_path": db_path},
        ),
        Tool.from_yaml(
            str(tool_dir / "ingest_markdown_schedules.tool.yaml"),
            binding=ingest_markdown_schedules,
            extra_kwargs={"db_path": db_path, "timezone": timezone},
        ),
        Tool.from_yaml(
            str(tool_dir / "ingest_chat_schedules.tool.yaml"),
            binding=ingest_chat_schedules,
            extra_kwargs={"db_path": db_path, "timezone": timezone},
        ),
        Tool.from_yaml(
            str(tool_dir / "ingest_feishu_markdown_schedules.tool.yaml"),
            binding=ingest_feishu_markdown_schedules,
            extra_kwargs={"db_path": db_path, "timezone": timezone},
        ),
        Tool.from_yaml(
            str(tool_dir / "ingest_feishu_doc_schedules.tool.yaml"),
            binding=ingest_feishu_doc_schedules,
            extra_kwargs={
                "db_path": db_path,
                "timezone": timezone,
                "feishu_base_url": os.getenv("FEISHU_BASE_URL", DEFAULT_FEISHU_BASE_URL),
            },
        ),
    ]

    llm_config = LLMConfig(
        model=os.getenv("LLM_MODEL"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
        api_type=os.getenv("LLM_API_TYPE", "openai_chat_completion"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
    )

    after_model_hooks = None
    before_tool_hooks = None
    after_tool_hooks = None
    if trace_callback is not None:
        after_model_hooks, before_tool_hooks, after_tool_hooks = _build_trace_hooks(trace_callback)

    config = AgentConfig(
        name="exercise1_schedule_assistant",
        max_context_tokens=100000,
        system_prompt=str(base_dir / "prompts" / "system_prompt.md"),
        system_prompt_type="jinja",
        tool_call_mode="structured",
        llm_config=llm_config,
        tools=tools,
        after_model_hooks=after_model_hooks,
        before_tool_hooks=before_tool_hooks,
        after_tool_hooks=after_tool_hooks,
    )

    return Agent(config=config)


def _assert_env() -> None:
    required = ["LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing env vars: "
            + ", ".join(missing)
            + ". Please set them in shell or .env before running."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise-1 personal schedule assistant powered by NexAU")
    parser.add_argument("--message", type=str, default=None, help="Run one-shot message and exit")
    parser.add_argument("--timezone", type=str, default=os.getenv("SCHEDULE_TIMEZONE", DEFAULT_TIMEZONE))
    parser.add_argument("--db-path", type=str, default=os.getenv("SCHEDULE_DB_PATH", str(DEFAULT_DB_PATH)))
    args = parser.parse_args()

    _load_local_env(Path(__file__).parent.resolve())

    _assert_env()

    db_path = str(Path(args.db_path).expanduser().resolve())
    timezone = args.timezone

    init_db(db_path=db_path)
    agent = _build_agent(base_dir=Path(__file__).parent.resolve(), timezone=timezone, db_path=db_path)

    def context() -> dict[str, str]:
        return {
            "date": _now_string(timezone),
            "timezone": timezone,
            "db_path": db_path,
        }

    if args.message:
        result = agent.run(message=args.message, context=context())
        print(result)
        return

    print("Exercise 1 - NexAU Personal Schedule Assistant")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        user_input = input("\nYou> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        result = agent.run(message=user_input, context=context())
        print(f"\nAgent> {result}")


if __name__ == "__main__":
    main()
