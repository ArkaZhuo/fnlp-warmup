from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import termios
import tty
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exercise1_nexau import scheduler_tools as st
from exercise1_nexau.run_scheduler_agent import _build_agent, _load_local_env

APP_TITLE = "Exercise 1 Schedule Agent"
DEFAULT_DB_PATH = Path(__file__).parent / "data" / "terminal_demo_schedule.db"
DEFAULT_FEISHU_URL = "https://wcnmcvjcsxin.feishu.cn/wiki/GpwnwG2UQiZ0bukNKDLcqoHvntd"
DEFAULT_CHAT_PATH = "./exercise1_nexau/data/sample_chat.txt"
DEFAULT_MARKDOWN_PATH = "./exercise1_nexau/data/sample_notes.md"
ANSI_RESET = "\033[0m"
ANSI_DIM = "\033[2m"
ANSI_FAINT = "\033[90m"
ANSI_SOFT = "\033[37m"
ANSI_BOLD = "\033[1m"
SELECTED_PREFIX = "›"
UNSELECTED_PREFIX = " "
_TRACE_STEP = 0
_TRACE_TOOL_USED_IN_TURN = False


def _payload(response: dict[str, str]) -> dict[str, Any]:
    return json.loads(response["content"])


def _now_string(timezone: str) -> str:
    tz = ZoneInfo(timezone)
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")


def _rule(char: str = "-") -> None:
    print(char * 72)


def _header(title: str) -> None:
    print()
    _rule("=")
    print(f"  {title}")
    _rule("=")


def _section(title: str) -> None:
    print()
    _rule()
    print(f"  {title}")
    _rule()


def _style(text: str, *codes: str) -> str:
    if not _supports_inline_select():
        return text
    prefix = "".join(codes)
    return f"{prefix}{text}{ANSI_RESET}"


def _trace(message: str) -> None:
    global _TRACE_TOOL_USED_IN_TURN
    global _TRACE_STEP
    _TRACE_STEP += 1
    if message.startswith("正在调用工具 ") or message.startswith("工具 "):
        _TRACE_TOOL_USED_IN_TURN = True
    label = _style(f"step {_TRACE_STEP:02d}", ANSI_FAINT)
    print(f"{label} {_style(message, ANSI_DIM)}")


def _reset_trace_turn() -> None:
    global _TRACE_TOOL_USED_IN_TURN
    _TRACE_TOOL_USED_IN_TURN = False


def _trace_used_tool_in_turn() -> bool:
    return _TRACE_TOOL_USED_IN_TURN


def _trace_tool_start(tool_name: str, detail: str | None = None) -> None:
    if detail:
        _trace(f"正在调用工具 {tool_name} | {detail}")
    else:
        _trace(f"正在调用工具 {tool_name}")


def _trace_tool_end(tool_name: str, payload: dict[str, Any]) -> None:
    if payload.get("ok") is False:
        detail = payload.get("error") or payload.get("message") or "执行失败"
        _trace(f"工具 {tool_name} 失败 | {detail}")
        return

    event = payload.get("event")
    if isinstance(event, dict):
        title = event.get("title", "")
        event_id = event.get("id", "")
        _trace(f"工具 {tool_name} 完成 | title={title}, event_id={event_id}")
        return

    if any(key in payload for key in ("created_count", "updated_count", "skipped_count")):
        _trace(
            f"工具 {tool_name} 完成 | 新增={payload.get('created_count', 0)}, "
            f"更新={payload.get('updated_count', 0)}, 跳过={payload.get('skipped_count', 0)}"
        )
        return

    if "count" in payload:
        _trace(f"工具 {tool_name} 完成 | count={payload.get('count', 0)}")
        return

    _trace(f"工具 {tool_name} 完成")


def _clean_input(value: str | None, default: str | None = None) -> str:
    cleaned = _sanitize_terminal_text((value or "").strip())
    if cleaned:
        return cleaned
    return default or ""


def _sanitize_terminal_text(value: str) -> str:
    """Recover or drop surrogate-escaped bytes produced by some terminals/IMEs."""
    if not value:
        return value
    try:
        recovered = value.encode("utf-8", "surrogateescape").decode("utf-8", "ignore")
    except Exception:
        recovered = value.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return recovered.replace("\x00", "").strip()


def _prompt_line(message: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"> {message}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default or ""
    cleaned = _sanitize_terminal_text(value)
    return cleaned or (default or "")


def _prompt_multiline(message: str) -> str:
    print(f"> {message}")
    print("  输入内容后，用单独一行 /done 结束。")
    lines: list[str] = []
    while True:
        try:
            line = input("  | ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip() == "/done":
            break
        lines.append(_sanitize_terminal_text(line))
    return _sanitize_terminal_text("\n".join(lines).strip())


def _write_temp_text(content: str, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=suffix,
        delete=False,
    )
    try:
        tmp.write(content)
        return tmp.name
    finally:
        tmp.close()


def _supports_inline_select() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _erase_last_lines(count: int) -> None:
    if count <= 0 or not _supports_inline_select():
        return
    for _ in range(count):
        sys.stdout.write("\x1b[F")
        sys.stdout.write("\x1b[2K")
    sys.stdout.write("\r")
    sys.stdout.flush()


def _read_key() -> str:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        first = sys.stdin.read(1)
        if first == "\x03":
            raise KeyboardInterrupt
        if first != "\x1b":
            return first
        second = sys.stdin.read(1)
        if second != "[":
            return first + second
        third = sys.stdin.read(1)
        return first + second + third
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _render_select_lines(
    title: str,
    values: list[tuple[str, str]],
    selected_index: int,
    allow_quit: bool,
) -> list[str]:
    hint = "使用 ↑ ↓ 选择，Enter 确认"
    if allow_quit:
        hint += "，q 返回"
    lines = [
        _style(title, ANSI_SOFT),
        _style(hint, ANSI_FAINT),
    ]
    for index, (_, label) in enumerate(values):
        if index == selected_index:
            lines.append(_style(f"{SELECTED_PREFIX} {label}", ANSI_BOLD))
        else:
            lines.append(_style(f"{UNSELECTED_PREFIX} {label}", ANSI_FAINT))
    return lines


def _select(
    title: str,
    values: list[tuple[str, str]],
    default: str | None = None,
    allow_quit: bool = True,
) -> str | None:
    if not values:
        return None

    if not _supports_inline_select():
        return default or values[0][0]

    selected_index = 0
    if default is not None:
        for index, (value, _) in enumerate(values):
            if value == default:
                selected_index = index
                break

    rendered_lines = 0
    try:
        while True:
            lines = _render_select_lines(title, values, selected_index, allow_quit)
            for line in lines:
                print(line)
            rendered_lines = len(lines)

            key = _read_key()
            if key in {"\r", "\n"}:
                selected_value, selected_label = values[selected_index]
                _erase_last_lines(rendered_lines)
                print(f"> {title}: {selected_label}")
                return selected_value
            if allow_quit and key.lower() == "q":
                _erase_last_lines(rendered_lines)
                print(f"> {title}: 返回")
                return None
            if key == "\x1b[A":
                selected_index = (selected_index - 1) % len(values)
            elif key == "\x1b[B":
                selected_index = (selected_index + 1) % len(values)

            _erase_last_lines(rendered_lines)
    except KeyboardInterrupt:
        _erase_last_lines(rendered_lines)
        print()
        return None


def _confirm(message: str, default: bool = False) -> bool:
    choice = _select(
        title=message,
        values=[("yes", "是"), ("no", "否")],
        default="yes" if default else "no",
        allow_quit=True,
    )
    return choice == "yes"


def _print_ingest_result(payload: dict[str, Any]) -> None:
    if not payload.get("ok"):
        print(f"同步失败：{payload.get('error')}")
        if payload.get("message"):
            print(f"说明：{payload.get('message')}")
        return

    print("同步完成")
    print(f"  来源：{payload.get('source_type')}")
    print(f"  模式：{payload.get('mode')}")
    print(f"  新增：{payload.get('created_count', 0)}")
    print(f"  更新：{payload.get('updated_count', 0)}")
    print(f"  跳过：{payload.get('skipped_count', 0)}")
    print(f"  冲突：{payload.get('conflicts_total', 0)}")

    created_ids = payload.get("created_ids") or []
    updated_ids = payload.get("updated_ids") or []
    if created_ids:
        print("  新增 event_id：")
        for event_id in created_ids:
            print(f"    - {event_id}")
    if updated_ids:
        print("  更新 event_id：")
        for event_id in updated_ids:
            print(f"    - {event_id}")

    sync_info = payload.get("feishu_calendar_sync") or {}
    if sync_info.get("enabled"):
        print(f"  飞书日历新增：{sync_info.get('synced_created_count', 0)}")
        print(f"  飞书日历更新：{sync_info.get('synced_updated_count', 0)}")
        if sync_info.get("errors"):
            print(f"  飞书日历错误：{len(sync_info['errors'])} 条")


def _format_event_line(index: int, event: dict[str, Any]) -> str:
    return (
        f"{index:>2}. {event['title']} | {event['start_time']} -> {event['end_time']} "
        f"| {event['source_type']} | {event['status']}"
    )


def _agent_context(timezone: str, db_path: str) -> dict[str, str]:
    return {
        "date": _now_string(timezone),
        "timezone": timezone,
        "db_path": db_path,
    }


def _build_chat_agent(db_path: str, timezone: str) -> Any | None:
    try:
        return _build_agent(
            base_dir=Path(__file__).parent.resolve(),
            timezone=timezone,
            db_path=db_path,
            trace_callback=_trace,
        )
    except Exception as exc:
        print(f"Agent 初始化失败：{exc}")
        return None


def _calendar_options() -> tuple[bool, str | None]:
    sync = _confirm("是否同步到飞书真实日历？", default=False)
    if not sync:
        return False, None
    calendar_id = _prompt_line("飞书日历 ID", os.getenv("FEISHU_CALENDAR_ID", "primary"))
    return True, calendar_id


def query_events(db_path: str, timezone: str) -> list[dict[str, Any]]:
    keyword = _prompt_line("关键词过滤，直接回车显示全部")
    _trace_tool_start("schedule_query", f"keyword={keyword or '全部'}")
    response = st.schedule_query(
        keyword=keyword or None,
        timezone=timezone,
        include_cancelled=True,
        limit=100,
        db_path=db_path,
    )
    payload = _payload(response)
    _trace_tool_end("schedule_query", payload)
    events = payload.get("events") or []
    print()
    if not events:
        print("暂无日程。")
        return []

    print(f"共 {len(events)} 条日程：")
    for index, event in enumerate(events, start=1):
        print(_format_event_line(index, event))
    return events


def choose_event(db_path: str, timezone: str) -> dict[str, Any] | None:
    events = query_events(db_path, timezone)
    if not events:
        return None

    values = [
        (
            event["id"],
            f"{event['title']} | {event['start_time']} -> {event['end_time']} | {event['status']}",
        )
        for event in events
    ]
    selected = _select(
        title="选择日程",
        values=values,
        allow_quit=True,
    )
    if not selected:
        return None
    for event in events:
        if event["id"] == selected:
            return event
    return None


def create_event(db_path: str, timezone: str) -> None:
    _section("手动创建日程")
    title = _prompt_line("标题", "项目复盘会议")
    start_time = _prompt_line("开始时间", "2026-04-18 10:00")
    end_time = _prompt_line("结束时间", "2026-04-18 11:00")
    description = _prompt_line("说明")
    sync, calendar_id = _calendar_options()
    _trace_tool_start("schedule_create", f"title={title}")
    response = st.schedule_create(
        title=title,
        start_time=start_time,
        end_time=end_time,
        timezone=timezone,
        description=description or None,
        sync_feishu_calendar=sync,
        feishu_calendar_id=calendar_id,
        db_path=db_path,
    )
    payload = _payload(response)
    _trace_tool_end("schedule_create", payload)
    if payload.get("ok"):
        event = payload["event"]
        print(f"创建成功：{event['title']} | event_id={event['id']}")
    else:
        print(f"创建失败：{payload.get('error')}")


def update_event(db_path: str, timezone: str) -> None:
    _section("修改日程")
    event = choose_event(db_path, timezone)
    if not event:
        return

    title = _prompt_line("标题", event["title"])
    start_time = _prompt_line("开始时间", event["start_time"].replace("T", " ")[:16])
    end_time = _prompt_line("结束时间", event["end_time"].replace("T", " ")[:16])
    description = _prompt_line("说明", event.get("description") or "")
    sync, calendar_id = _calendar_options()
    _trace_tool_start("schedule_update", f"event_id={event['id']}")
    response = st.schedule_update(
        event_id=event["id"],
        title=title,
        start_time=start_time,
        end_time=end_time,
        timezone=timezone,
        description=description or None,
        sync_feishu_calendar=sync,
        feishu_calendar_id=calendar_id,
        db_path=db_path,
    )
    payload = _payload(response)
    _trace_tool_end("schedule_update", payload)
    if payload.get("ok"):
        updated = payload["event"]
        print(f"修改成功：{updated['title']} | event_id={updated['id']}")
    else:
        print(f"修改失败：{payload.get('error')}")


def cancel_event(db_path: str, timezone: str) -> None:
    _section("取消日程")
    event = choose_event(db_path, timezone)
    if not event:
        return

    confirmed = _confirm(f"确认取消「{event['title']}」？", default=False)
    if not confirmed:
        print("已放弃取消。")
        return

    sync, calendar_id = _calendar_options()
    _trace_tool_start("schedule_delete", f"event_id={event['id']}")
    response = st.schedule_delete(
        event_id=event["id"],
        confirm=True,
        sync_feishu_calendar=sync,
        feishu_calendar_id=calendar_id,
        db_path=db_path,
    )
    payload = _payload(response)
    _trace_tool_end("schedule_delete", payload)
    if payload.get("ok"):
        print(f"已取消：{payload['event']['title']} | event_id={payload['event']['id']}")
    else:
        print(f"取消失败：{payload.get('error')}")


def _quick_actions(db_path: str, timezone: str) -> None:
    while True:
        action = _select(
            title="后续操作",
            values=[
                ("back", "返回"),
                ("view", "查看日程"),
                ("update", "修改日程"),
                ("cancel", "取消日程"),
            ],
            default="back",
            allow_quit=True,
        )
        if not action or action == "back":
            return
        if action == "view":
            query_events(db_path, timezone)
        elif action == "update":
            update_event(db_path, timezone)
        elif action == "cancel":
            cancel_event(db_path, timezone)


def ingest_chat_file(path: str, db_path: str, timezone: str, mode: str = "upsert") -> None:
    sync, calendar_id = _calendar_options()
    _trace_tool_start("ingest_chat_schedules", f"path={path}, mode={mode}")
    response = st.ingest_chat_schedules(
        file_path=path,
        timezone=timezone,
        mode=mode,
        sync_feishu_calendar=sync,
        feishu_calendar_id=calendar_id,
        db_path=db_path,
    )
    payload = _payload(response)
    _trace_tool_end("ingest_chat_schedules", payload)
    _print_ingest_result(payload)


def _chat_conversation_loop(agent: Any, db_path: str, timezone: str) -> None:
    _section("对话模式")
    print("现在进入连续对话。")
    print("支持三种输入方式：")
    print("1. 直接输入一句自然语言，让 Agent 判断创建 / 修改 / 查询。")
    print("2. 直接输入一个本地聊天记录文件路径，系统会按聊天记录导入。")
    print("3. 输入 /paste 后直接粘贴多行聊天记录。")
    print("输入 /view 查看日程，输入 /back 返回。")

    while True:
        user_input = _sanitize_terminal_text(_prompt_line("You"))
        if not user_input:
            continue

        command = user_input.strip().lower()
        if command in {"/back", "back", "返回", "quit", "exit"}:
            return
        if command in {"/view", "view", "查看"}:
            query_events(db_path, timezone)
            continue

        if command in {"/paste", "paste"}:
            content = _prompt_multiline("请粘贴聊天记录")
            if not content:
                print("内容为空，已取消。")
                continue
            mode = _select(
                title="同步模式",
                values=[
                    ("upsert", "upsert: 同名同日更新，否则创建"),
                    ("create", "create: 一律新建"),
                ],
                default="upsert",
                allow_quit=False,
            ) or "upsert"
            path = _write_temp_text(content, ".txt")
            ingest_chat_file(path, db_path, timezone, mode=mode)
            print(_style("已处理聊天记录。你可以继续输入下一条内容，或输入 /view 查看日程。", ANSI_FAINT))
            continue

        candidate_path = Path(user_input).expanduser()
        if candidate_path.exists() and candidate_path.is_file():
            mode = _select(
                title="同步模式",
                values=[
                    ("upsert", "upsert: 同名同日更新，否则创建"),
                    ("create", "create: 一律新建"),
                ],
                default="upsert",
                allow_quit=False,
            ) or "upsert"
            ingest_chat_file(str(candidate_path.resolve()), db_path, timezone, mode=mode)
            print(_style("已处理聊天记录文件。你可以继续输入下一条内容，或输入 /view 查看日程。", ANSI_FAINT))
            continue

        try:
            _reset_trace_turn()
            _trace("Agent 正在分析用户输入")
            result = agent.run(message=user_input, context=_agent_context(timezone, db_path))
            print(f"\nAgent> {result}")
        except Exception as exc:
            error_text = _sanitize_terminal_text(str(exc))
            if "surrogates not allowed" in error_text or "codec can't encode character" in error_text:
                print("Agent 执行失败：终端输入里含有异常编码字符。请重新输入一次，或切换英文输入法后再输入。")
            else:
                print(f"Agent 执行失败：{error_text}")
            continue

        if _trace_used_tool_in_turn():
            print(_style("Agent 已执行工具。你可以继续输入下一条需求，或输入 /view 查看日程。", ANSI_FAINT))
        else:
            print(_style("Agent 正在等待你补充信息。直接继续输入回答即可。", ANSI_FAINT))


def chat_agent_loop(db_path: str, timezone: str) -> None:
    _header("从聊天中创建 / 修改日程")
    print("这里把自然语言、聊天记录文件、粘贴聊天记录统一收进“输入内容”。")

    agent = _build_chat_agent(db_path, timezone)
    if agent is None:
        return

    while True:
        action = _select(
            title="聊天模式",
            values=[
                ("input", "输入内容"),
                ("view", "查看当前日程"),
                ("back", "返回"),
            ],
            default="input",
            allow_quit=True,
        )

        if not action or action == "back":
            return

        if action == "view":
            query_events(db_path, timezone)
            continue

        _chat_conversation_loop(agent, db_path, timezone)


def ingest_from_feishu(db_path: str, timezone: str) -> None:
    _header("从飞书文档创建 / 修改日程")
    url = _prompt_line("飞书 docx/wiki 链接", DEFAULT_FEISHU_URL)
    mode = _select(
        title="同步模式",
        values=[
            ("upsert", "upsert: 同名同日更新，否则创建"),
            ("create", "create: 一律新建"),
        ],
        default="upsert",
        allow_quit=False,
    ) or "upsert"
    sync, calendar_id = _calendar_options()
    _trace_tool_start("ingest_feishu_doc_schedules", f"mode={mode}")
    response = st.ingest_feishu_doc_schedules(
        document_id_or_url=url,
        timezone=timezone,
        mode=mode,
        sync_feishu_calendar=sync,
        feishu_calendar_id=calendar_id,
        db_path=db_path,
    )
    payload = _payload(response)
    _trace_tool_end("ingest_feishu_doc_schedules", payload)
    _print_ingest_result(payload)
    _quick_actions(db_path, timezone)


def ingest_from_markdown(db_path: str, timezone: str) -> None:
    _header("从 Markdown 笔记创建 / 修改日程")
    choice = _select(
        title="Markdown 输入",
        values=[
            ("file", "输入 Markdown 文件路径"),
            ("paste", "直接粘贴 Markdown 内容"),
        ],
        default="file",
        allow_quit=True,
    )
    if not choice:
        return

    if choice == "paste":
        content = _prompt_multiline("请粘贴 Markdown")
        if not content:
            print("内容为空，已取消。")
            return
        path = _write_temp_text(content, ".md")
    else:
        path = _prompt_line("Markdown 文件路径", DEFAULT_MARKDOWN_PATH)

    mode = _select(
        title="同步模式",
        values=[
            ("upsert", "upsert: 同名同日更新，否则创建"),
            ("create", "create: 一律新建"),
        ],
        default="upsert",
        allow_quit=False,
    ) or "upsert"
    sync, calendar_id = _calendar_options()
    _trace_tool_start("ingest_markdown_schedules", f"path={path}, mode={mode}")
    response = st.ingest_markdown_schedules(
        file_path=path,
        timezone=timezone,
        mode=mode,
        sync_feishu_calendar=sync,
        feishu_calendar_id=calendar_id,
        db_path=db_path,
    )
    payload = _payload(response)
    _trace_tool_end("ingest_markdown_schedules", payload)
    _print_ingest_result(payload)
    _quick_actions(db_path, timezone)


def run_demo(db_path: str, timezone: str) -> None:
    st.init_db(db_path=db_path)
    _header(APP_TITLE)
    print("这是练习一的终端后端 Demo。")
    print("支持聊天、飞书文档、Markdown 三种输入源。")
    print("菜单使用上下键，回车确认。")
    print(f"数据库：{db_path}")

    while True:
        choice = _select(
            title="选择输入源或操作",
            values=[
                ("chat", "从聊天中创建 / 修改日程"),
                ("feishu", "从飞书文档创建 / 修改日程"),
                ("markdown", "从 Markdown 笔记创建 / 修改日程"),
                ("view", "查看日程"),
                ("update", "修改日程"),
                ("cancel", "取消日程"),
                ("exit", "退出"),
            ],
            default="chat",
            allow_quit=False,
        )

        if choice == "exit":
            print("已退出。")
            break
        if choice == "chat":
            chat_agent_loop(db_path, timezone)
        elif choice == "feishu":
            ingest_from_feishu(db_path, timezone)
        elif choice == "markdown":
            ingest_from_markdown(db_path, timezone)
        elif choice == "view":
            query_events(db_path, timezone)
        elif choice == "update":
            update_event(db_path, timezone)
        elif choice == "cancel":
            cancel_event(db_path, timezone)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise1 terminal schedule agent demo")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--timezone", default=st.DEFAULT_TIMEZONE)
    args = parser.parse_args()

    _load_local_env(Path(__file__).parent.resolve())
    run_demo(
        db_path=str(Path(args.db_path).expanduser().resolve()),
        timezone=args.timezone,
    )


if __name__ == "__main__":
    main()
