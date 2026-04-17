from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from agent import PersonalScheduleAgent
from schemas import PendingAction


ROOT = Path(__file__).resolve().parent
AGENT = PersonalScheduleAgent(ROOT)
PENDING: dict[str, PendingAction | None] = {"default": None}


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Personal Schedule Agent</title>
  <style>
    :root {
      --bg: #f6efe3;
      --panel: rgba(255,250,242,0.95);
      --line: #d8c7af;
      --ink: #2f2924;
      --sub: #7e7062;
      --accent: #b85f34;
      --user: #f5dfca;
      --agent: #f1ebe2;
    }
    body {
      margin: 0;
      font-family: "Noto Serif SC", "Source Han Serif SC", serif;
      background:
        radial-gradient(circle at 10% 10%, #fff7ea 0, transparent 22%),
        linear-gradient(180deg, #ede0ca 0%, var(--bg) 42%, #faf6ef 100%);
      color: var(--ink);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      padding: 24px;
      box-sizing: border-box;
    }
    .shell {
      width: min(1040px, 100%);
      height: min(800px, calc(100vh - 48px));
      display: grid;
      grid-template-columns: 1.3fr 0.8fr;
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 18px 54px rgba(91, 67, 43, 0.10);
      overflow: hidden;
      backdrop-filter: blur(8px);
    }
    .chat { display: grid; grid-template-rows: auto 1fr auto; }
    .header { padding: 18px 20px 14px; border-bottom: 1px solid rgba(196, 172, 140, 0.6); }
    .title { margin: 0; font-size: 24px; letter-spacing: 0.02em; }
    .subtitle { margin: 6px 0 0; color: var(--sub); font-size: 14px; }
    #messages { padding: 18px 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; }
    .bubble { border-radius: 16px; padding: 12px 14px; line-height: 1.6; white-space: pre-wrap; font-size: 14px; border: 1px solid rgba(157, 128, 94, 0.15); }
    .user { align-self: flex-end; background: var(--user); max-width: 74%; }
    .agent { align-self: flex-start; background: var(--agent); max-width: 86%; }
    .meta { color: var(--sub); font-size: 12px; margin-top: 8px; }
    .composer { border-top: 1px solid rgba(196, 172, 140, 0.6); padding: 14px; display: grid; grid-template-columns: 1fr auto; gap: 10px; }
    textarea { resize: none; min-height: 72px; border: 1px solid var(--line); background: #fffcf7; border-radius: 14px; padding: 12px 14px; font: inherit; color: inherit; outline: none; }
    button { align-self: end; border: none; border-radius: 999px; background: linear-gradient(135deg, #c06a3f, #9e4d26); color: white; padding: 12px 18px; font: inherit; cursor: pointer; }
    .side { display: grid; grid-template-rows: auto 1fr; }
    .side-body { padding: 18px 20px; overflow-y: auto; font-size: 14px; line-height: 1.6; }
    .card { background: rgba(255,255,255,0.45); border: 1px solid rgba(196, 172, 140, 0.55); border-radius: 14px; padding: 12px 14px; margin-bottom: 12px; }
    .card h4 { margin: 0 0 8px; font-size: 14px; }
    .tag { display: inline-block; font-size: 12px; background: #ead8bf; color: #5a4838; padding: 3px 8px; border-radius: 999px; margin-right: 6px; margin-bottom: 6px; }
    pre { white-space: pre-wrap; word-break: break-word; margin: 0; }
    .event-row { padding: 9px 0; border-bottom: 1px dashed rgba(196, 172, 140, 0.55); }
    .event-row:last-child { border-bottom: none; }
    .event-title { font-weight: 600; }
    .event-meta { color: var(--sub); font-size: 12px; margin-top: 4px; }
    @media (max-width: 860px) { .shell { grid-template-columns: 1fr; height: auto; } }
  </style>
</head>
<body>
  <div class="shell">
    <section class="panel chat">
      <div class="header">
        <h1 class="title">Personal Schedule Agent</h1>
        <p class="subtitle">支持聊天记录、Markdown 笔记、飞书导出文本检索，并支持缺参追问。</p>
      </div>
      <div id="messages"></div>
      <div class="composer">
        <textarea id="input" placeholder="例如：帮我把明天下午和李老师讨论论文的会加到日程里，大概两点，看看笔记里有没有具体会议室。"></textarea>
        <button id="send">发送</button>
      </div>
    </section>
    <aside class="panel side">
      <div class="header">
        <h2 class="title" style="font-size:20px">当前决策</h2>
        <p class="subtitle">展示检索命中、缺失字段与工具参数。</p>
      </div>
      <div id="state" class="side-body"></div>
    </aside>
  </div>
  <script>
    const messages = document.getElementById('messages');
    const state = document.getElementById('state');
    const input = document.getElementById('input');
    const sendBtn = document.getElementById('send');

    function bubble(role, text, meta='') {
      const div = document.createElement('div');
      div.className = 'bubble ' + role;
      div.textContent = text;
      if (meta) {
        const extra = document.createElement('div');
        extra.className = 'meta';
        extra.textContent = meta;
        div.appendChild(extra);
      }
      messages.appendChild(div);
      messages.scrollTop = messages.scrollHeight;
    }

    function renderState(payload) {
      const decision = payload.decision;
      const pending = payload.pending_action;
      const hits = decision.retrieved_hits || [];
      const events = payload.calendar_events || [];
      const tags = (decision.missing_fields || []).map(x => `<span class="tag">${x}</span>`).join('');
      state.innerHTML = `
        <div class="card"><h4>Intent</h4><div>${decision.intent}</div></div>
        <div class="card"><h4>缺失字段</h4><div>${tags || '无'}</div></div>
        <div class="card"><h4>工具调用</h4><div>${decision.tool_name || '尚未执行工具'}</div><pre>${JSON.stringify(decision.tool_args || {}, null, 2)}</pre></div>
        <div class="card"><h4>待补上下文</h4><pre>${JSON.stringify(pending || null, null, 2)}</pre></div>
        <div class="card"><h4>检索命中</h4>${
          hits.length ? hits.map(hit => `<div style="margin-bottom:10px;"><div><span class="tag">${hit.source_type}</span> ${hit.source_name}</div><div>${hit.snippet}</div></div>`).join('') : '无'
        }</div>
        <div class="card"><h4>当前日程列表</h4>${
          events.length ? events.map(event => `
            <div class="event-row">
              <div class="event-title">${event.title}</div>
              <div class="event-meta">${event.date} ${event.start_time}${event.location ? ` @ ${event.location}` : ''}</div>
            </div>
          `).join('') : '暂无日程'
        }</div>
      `;
    }

    async function send() {
      const text = input.value.trim();
      if (!text) return;
      bubble('user', text);
      input.value = '';
      const resp = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: new URLSearchParams({message: text}),
      });
      const payload = await resp.json();
      bubble('agent', payload.decision.response, payload.decision.tool_name ? `tool: ${payload.decision.tool_name}` : 'tool: none');
      renderState(payload);
    }

    sendBtn.addEventListener('click', send);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    });

    bubble('agent', '你好，我可以帮你新增日程、查询日程、修改日程。如果信息不完整，我会继续追问。');
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in {"/", "/index.html"}:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/chat":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        message = parse_qs(body).get("message", [""])[0].strip()

        session_id = "default"
        result = AGENT.handle(message, pending_action=PENDING.get(session_id))
        PENDING[session_id] = result.pending_action
        payload = {
            "decision": {
                "intent": result.decision.intent,
                "missing_fields": result.decision.missing_fields,
                "tool_name": result.decision.tool_name,
                "tool_args": result.decision.tool_args,
                "retrieved_hits": [
                    {
                        "source_type": hit.source_type,
                        "source_name": hit.source_name,
                        "score": round(hit.score, 2),
                        "snippet": hit.snippet,
                    }
                    for hit in result.decision.retrieved_hits
                ],
                "response": result.decision.response,
            },
            "pending_action": {
                "intent": result.pending_action.intent,
                "missing_fields": result.pending_action.missing_fields,
                "collected_fields": result.pending_action.collected_fields,
            }
            if result.pending_action is not None
            else None,
            "calendar_events": [
                {
                    "event_id": event.event_id,
                    "title": event.title,
                    "date": event.date,
                    "start_time": event.start_time,
                    "location": event.location,
                }
                for event in AGENT.calendar.upcoming_events()
            ],
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def main() -> None:
    server = HTTPServer(("127.0.0.1", 8008), Handler)
    print("Web UI 已启动: http://127.0.0.1:8008")
    server.serve_forever()


if __name__ == "__main__":
    main()
