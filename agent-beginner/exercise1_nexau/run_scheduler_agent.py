"""Run exercise-1 personal schedule assistant with NexAU.

Usage:
  python run_scheduler_agent.py --message "明天下午3点安排项目复盘"
  python run_scheduler_agent.py
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from nexau import Agent, AgentConfig, LLMConfig, Tool

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


def _now_string(timezone: str) -> str:
    tz = ZoneInfo(timezone)
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")


def _build_agent(base_dir: Path, timezone: str, db_path: str) -> Agent:
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

    config = AgentConfig(
        name="exercise1_schedule_assistant",
        max_context_tokens=100000,
        system_prompt=str(base_dir / "prompts" / "system_prompt.md"),
        system_prompt_type="jinja",
        tool_call_mode="structured",
        llm_config=llm_config,
        tools=tools,
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

    if load_dotenv is not None:
        load_dotenv()

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
