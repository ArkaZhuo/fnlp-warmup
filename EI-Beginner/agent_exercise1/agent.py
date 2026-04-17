from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

from retrievers import MultiSourceRetriever
from schemas import AgentDecision, CalendarEvent, PendingAction, RetrievalHit, TurnResult
from tools import CalendarStore, make_event_id


def resolve_relative_date(text: str, today: dt.date | None = None) -> str | None:
    today = today or dt.date.today()
    if "明天" in text:
        return (today + dt.timedelta(days=1)).isoformat()
    if "后天" in text:
        return (today + dt.timedelta(days=2)).isoformat()
    return None


def extract_time(text: str) -> str | None:
    if "两点" in text or "2点" in text:
        return "14:00" if "下午" in text else "02:00"
    if "三点" in text or "3点" in text:
        return "15:00" if "下午" in text else "03:00"
    if "四点" in text or "4点" in text:
        return "16:00" if "下午" in text else "04:00"
    m = re.search(r"(\d{1,2}):(\d{2})", text)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    return None


def extract_people(text: str) -> list[str]:
    if "导师" in text:
        return ["导师"]

    candidates = re.findall(r"([A-Za-z0-9\u4e00-\u9fff]{1,6}(?:老师|导师|同学))", text)
    cleaned = []
    for item in candidates:
        for prefix in ["帮我安排和", "帮我和", "我和", "安排和", "和", "与", "把", "跟"]:
            if item.startswith(prefix):
                item = item[len(prefix) :]
        item = re.sub(r"^(?:明天|后天|今天|上午|下午|晚上|和|与|把)+", "", item)
        item = re.sub(r"(讨论|开会|开题|论文).*?$", "", item)
        item = item.strip()
        if item:
            cleaned.append(item)
    return list(dict.fromkeys(cleaned))


def extract_title(text: str, people: list[str]) -> str:
    lead = "、".join(people) if people else "相关人员"
    if "论文" in text:
        return f"与{lead}讨论论文"
    if "开题" in text:
        return f"与{lead}讨论开题"
    if "组会" in text:
        return "组会"
    if "导师" in people:
        return "与导师会面"
    return f"与{lead}会面"


def extract_location_from_hits(hits: list[RetrievalHit]) -> str | None:
    pattern = re.compile(
        r"(光华楼(?:东辅楼)?[A-Za-z0-9\u4e00-\u9fff]*\d{0,3}|逸夫楼[A-Za-z0-9\u4e00-\u9fff]*\d{0,3}|东辅楼[A-Za-z0-9\u4e00-\u9fff]*\d{0,3}|线上会议室|腾讯会议)"
    )

    def normalize(candidate: str) -> str:
        candidate = candidate.strip("，。；;:：!?！？")
        candidate = re.sub(r"(吗|呢|吧)$", "", candidate)
        return candidate.strip()

    candidates: list[str] = []
    for hit in hits:
        path = hit.metadata.get("path")
        if path:
            text = Path(path).read_text(encoding="utf-8")
            candidates.extend(normalize(m.group(1)) for m in pattern.finditer(text))
        candidates.extend(normalize(m.group(1)) for m in pattern.finditer(hit.snippet))

    if not candidates:
        return None

    def location_score(loc: str) -> tuple[int, int]:
        has_room_number = int(bool(re.search(r"\d{2,3}$", loc)))
        has_sub_building = int(any(token in loc for token in ["东辅楼", "会议室"]))
        return (has_room_number + has_sub_building, len(loc))

    candidates = [loc for loc in candidates if loc]
    candidates.sort(key=location_score, reverse=True)
    return candidates[0] if candidates else None


def build_retrieval_queries(text: str) -> list[str]:
    queries = [text]
    people = extract_people(text)
    for person in people:
        queries.append(person)
        if "论文" in text:
            queries.append(f"{person} 论文")
        if "开题" in text:
            queries.append(f"{person} 开题")
    for keyword in ["论文", "开题", "组会", "会议室", "光华楼", "逸夫楼", "导师", "老师"]:
        if keyword in text:
            queries.append(keyword)
    return list(dict.fromkeys(q for q in queries if q.strip()))


def field_display_name(field_name: str) -> str:
    mapping = {
        "date": "日期",
        "start_time": "开始时间",
        "target_event": "要修改的日程对象",
        "new_start_time": "新的开始时间",
    }
    return mapping.get(field_name, field_name)


def parse_partial_fields(text: str) -> dict[str, str]:
    data: dict[str, str] = {}
    date = resolve_relative_date(text)
    start_time = extract_time(text)
    people = extract_people(text)
    if date:
        data["date"] = date
    if start_time:
        data["start_time"] = start_time
        data["new_start_time"] = start_time
    if people:
        data["target_event"] = people[0]
    elif "导师" in text:
        data["target_event"] = "导师"
    elif "组会" in text:
        data["target_event"] = "组会"
    return data


class PersonalScheduleAgent:
    def __init__(self, root: Path):
        self.root = root
        self.retriever = MultiSourceRetriever(root / "data")
        self.calendar = CalendarStore(root / "results" / "calendar_events.json")

    def decide_intent(self, text: str) -> str:
        if any(k in text for k in ["加到日程", "加入日程", "安排一下", "加个日程", "新增日程", "加到日历"]):
            return "add_schedule"
        if ("安排" in text or "加" in text) and any(k in text for k in ["会", "会议", "组会", "讨论", "导师", "老师", "同学"]):
            return "add_schedule"
        if any(k in text for k in ["延后", "推迟", "改到", "改成", "修改"]):
            return "modify_schedule"
        if any(k in text for k in ["有什么安排", "查看日程", "查一下日程", "最近安排"]):
            return "list_schedule"
        return "unknown"

    def handle(self, text: str, pending_action: PendingAction | None = None) -> TurnResult:
        if pending_action is not None:
            return self._handle_followup(text, pending_action)

        intent = self.decide_intent(text)
        hits = self.retriever.search_multi_query(build_retrieval_queries(text))

        if intent == "add_schedule":
            return self._handle_add(text, hits)
        if intent == "modify_schedule":
            return self._handle_modify(text, hits)
        if intent == "list_schedule":
            return self._handle_list(text, hits)

        return TurnResult(
            decision=AgentDecision(
                user_input=text,
                intent=intent,
                missing_fields=[],
                retrieved_hits=hits,
                tool_name=None,
                tool_args={},
                response="我目前只支持新增日程、修改日程和查询日程三类请求。",
            ),
            pending_action=None,
        )

    def _handle_add(self, text: str, hits: list[RetrievalHit]) -> TurnResult:
        date = resolve_relative_date(text)
        start_time = extract_time(text)
        participants = extract_people(text)
        title = extract_title(text, participants)
        location = extract_location_from_hits(hits)

        missing = []
        if date is None:
            missing.append("date")
        if start_time is None:
            missing.append("start_time")

        if missing:
            pending = PendingAction(
                intent="add_schedule",
                original_user_input=text,
                collected_fields={
                    "participants": participants,
                    "title": title,
                    "location": location,
                    "source": [hit.source_type for hit in hits[:3]],
                },
                missing_fields=missing,
            )
            names = "、".join(field_display_name(name) for name in missing)
            return TurnResult(
                decision=AgentDecision(
                    user_input=text,
                    intent="add_schedule",
                    missing_fields=missing,
                    retrieved_hits=hits,
                    tool_name=None,
                    tool_args={},
                    response=f"我已经识别出这是新增日程请求，但还缺少：{names}。请直接补充这些信息，我会继续完成。",
                ),
                pending_action=pending,
            )

        event = CalendarEvent(
            event_id=make_event_id(),
            title=title,
            date=date,
            start_time=start_time,
            location=location,
            participants=participants,
            description=text,
            source=[hit.source_type for hit in hits[:3]],
        )
        self.calendar.add_event(event)
        return TurnResult(
            decision=AgentDecision(
                user_input=text,
                intent="add_schedule",
                missing_fields=[],
                retrieved_hits=hits,
                tool_name="add_schedule",
                tool_args=event.to_dict(),
                response=f"已添加日程：{event.title}，时间 {event.date} {event.start_time}" + (
                    f"，地点 {event.location}" if event.location else ""
                ),
            ),
            pending_action=None,
        )

    def _handle_modify(self, text: str, hits: list[RetrievalHit]) -> TurnResult:
        new_time = extract_time(text)
        keyword = ""
        people = extract_people(text)
        if people:
            keyword = people[0]
        elif "导师" in text:
            keyword = "导师"
        if "组会" in text:
            keyword = "组会"

        missing = []
        if not keyword:
            missing.append("target_event")
        if not new_time:
            missing.append("new_start_time")

        if missing:
            pending = PendingAction(
                intent="modify_schedule",
                original_user_input=text,
                collected_fields={},
                missing_fields=missing,
            )
            names = "、".join(field_display_name(name) for name in missing)
            return TurnResult(
                decision=AgentDecision(
                    user_input=text,
                    intent="modify_schedule",
                    missing_fields=missing,
                    retrieved_hits=hits,
                    tool_name=None,
                    tool_args={},
                    response=f"我判断你想修改日程，但还缺少：{names}。请继续补充。",
                ),
                pending_action=pending,
            )

        event = self.calendar.find_event(keyword)
        if event is None:
            return TurnResult(
                decision=AgentDecision(
                    user_input=text,
                    intent="modify_schedule",
                    missing_fields=[],
                    retrieved_hits=hits,
                    tool_name=None,
                    tool_args={"keyword": keyword},
                    response=f"我没有在现有日程中定位到与“{keyword}”对应的唯一事件，暂时无法修改。",
                ),
                pending_action=None,
            )

        updated = self.calendar.modify_event_time(event.event_id, new_time)
        return TurnResult(
            decision=AgentDecision(
                user_input=text,
                intent="modify_schedule",
                missing_fields=[],
                retrieved_hits=hits,
                tool_name="modify_schedule",
                tool_args={"event_id": event.event_id, "new_start_time": new_time},
                response=f"已将“{updated.title}”调整到 {updated.date} {updated.start_time}。",
            ),
            pending_action=None,
        )

    def _handle_list(self, text: str, hits: list[RetrievalHit]) -> TurnResult:
        events = self.calendar.list_events()
        date = resolve_relative_date(text)
        if date:
            events = [event for event in events if event.date == date]

        if not events:
            response = "当前没有查到符合条件的日程。"
        else:
            lines = []
            for event in events:
                line = f"- {event.date} {event.start_time} {event.title}"
                if event.location:
                    line += f" @ {event.location}"
                lines.append(line)
            response = "查到以下日程：\n" + "\n".join(lines)

        return TurnResult(
            decision=AgentDecision(
                user_input=text,
                intent="list_schedule",
                missing_fields=[],
                retrieved_hits=hits,
                tool_name="list_schedule",
                tool_args={"date": date} if date else {},
                response=response,
            ),
            pending_action=None,
        )

    def _handle_followup(self, text: str, pending_action: PendingAction) -> TurnResult:
        partial = parse_partial_fields(text)
        collected = dict(pending_action.collected_fields)
        collected.update(partial)
        missing = [name for name in pending_action.missing_fields if name not in collected]
        merged_text = pending_action.original_user_input + "\n补充信息：" + text
        hits = self.retriever.search_multi_query(build_retrieval_queries(merged_text))

        if pending_action.intent == "add_schedule":
            if "location" not in collected or not collected.get("location"):
                collected["location"] = extract_location_from_hits(hits)
            if missing:
                names = "、".join(field_display_name(name) for name in missing)
                return TurnResult(
                    decision=AgentDecision(
                        user_input=text,
                        intent="add_schedule",
                        missing_fields=missing,
                        retrieved_hits=hits,
                        tool_name=None,
                        tool_args=collected,
                        response=f"我还缺少：{names}。请继续补充。",
                    ),
                    pending_action=PendingAction(
                        intent=pending_action.intent,
                        original_user_input=pending_action.original_user_input,
                        collected_fields=collected,
                        missing_fields=missing,
                    ),
                )

            event = CalendarEvent(
                event_id=make_event_id(),
                title=collected.get("title", "未命名日程"),
                date=collected["date"],
                start_time=collected["start_time"],
                location=collected.get("location"),
                participants=collected.get("participants", []),
                description=merged_text,
                source=collected.get("source", [hit.source_type for hit in hits[:3]]),
            )
            self.calendar.add_event(event)
            return TurnResult(
                decision=AgentDecision(
                    user_input=text,
                    intent="add_schedule",
                    missing_fields=[],
                    retrieved_hits=hits,
                    tool_name="add_schedule",
                    tool_args=event.to_dict(),
                    response=f"已根据补充信息添加日程：{event.title}，时间 {event.date} {event.start_time}" + (
                        f"，地点 {event.location}" if event.location else ""
                    ),
                ),
                pending_action=None,
            )

        if pending_action.intent == "modify_schedule":
            if missing:
                names = "、".join(field_display_name(name) for name in missing)
                return TurnResult(
                    decision=AgentDecision(
                        user_input=text,
                        intent="modify_schedule",
                        missing_fields=missing,
                        retrieved_hits=hits,
                        tool_name=None,
                        tool_args=collected,
                        response=f"我还缺少：{names}。请继续补充。",
                    ),
                    pending_action=PendingAction(
                        intent=pending_action.intent,
                        original_user_input=pending_action.original_user_input,
                        collected_fields=collected,
                        missing_fields=missing,
                    ),
                )

            event = self.calendar.find_event(collected["target_event"])
            if event is None:
                return TurnResult(
                    decision=AgentDecision(
                        user_input=text,
                        intent="modify_schedule",
                        missing_fields=[],
                        retrieved_hits=hits,
                        tool_name=None,
                        tool_args=collected,
                        response=f"我仍然没有定位到与“{collected['target_event']}”对应的唯一事件，无法修改。",
                    ),
                    pending_action=None,
                )
            updated = self.calendar.modify_event_time(event.event_id, collected["new_start_time"])
            return TurnResult(
                decision=AgentDecision(
                    user_input=text,
                    intent="modify_schedule",
                    missing_fields=[],
                    retrieved_hits=hits,
                    tool_name="modify_schedule",
                    tool_args={"event_id": event.event_id, "new_start_time": collected["new_start_time"]},
                    response=f"已根据补充信息将“{updated.title}”调整到 {updated.date} {updated.start_time}。",
                ),
                pending_action=None,
            )

        return TurnResult(
            decision=AgentDecision(
                user_input=text,
                intent="unknown",
                missing_fields=[],
                retrieved_hits=hits,
                tool_name=None,
                tool_args={},
                response="当前没有可继续执行的待办动作。",
            ),
            pending_action=None,
        )


def format_decision(decision: AgentDecision, pending_action: PendingAction | None = None) -> str:
    payload = {
        "intent": decision.intent,
        "missing_fields": decision.missing_fields,
        "tool_name": decision.tool_name,
        "tool_args": decision.tool_args,
        "retrieved_hits": [
            {
                "source_type": hit.source_type,
                "source_name": hit.source_name,
                "score": round(hit.score, 2),
                "snippet": hit.snippet,
            }
            for hit in decision.retrieved_hits
        ],
        "response": decision.response,
        "pending_action": {
            "intent": pending_action.intent,
            "missing_fields": pending_action.missing_fields,
            "collected_fields": pending_action.collected_fields,
        }
        if pending_action is not None
        else None,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
