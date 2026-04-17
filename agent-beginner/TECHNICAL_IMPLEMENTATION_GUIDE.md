# 智能体项目技术实现文档（基于当前仓库要求）

## 1. 文档目标
本文件基于当前仓库中唯一需求文件 `README.md` 编写，目标是给出一套可直接落地的实现方案，覆盖：

1. 练习一：个人日程助手（支持工具调用、从聊天记录与飞书文档/Markdown 笔记生成/修改日程）
2. 练习二：NexDR 的理解与改造（替换搜索、取消 HTML、支持 PDF/图片输入）
3. 思考题的工程化答案（MCP/Skills、GPU 利用率、反馈训练）

> 当前仓库暂无业务代码，只有需求说明。因此本文档同时包含“工程脚手架建议 + 详细实现细节”。

---

## 2. 当前代码现状

仓库当前文件：

- `README.md`：练习要求
- `.ipynb_checkpoints/README-checkpoint.md`：与 README 内容一致

没有可复用业务模块，因此建议按本文第 4、5 节先搭建项目结构。

---

## 3. 总体实施路线（建议）

采用“两条主线并行”的实施方式：

1. 主线 A：先做练习一，建立通用 Agent 能力底座（LLM 接入、工具系统、文档解析、检索、任务编排）
2. 主线 B：在主线 A 底座上完成练习二（NexDR 改造），复用检索与多模态输入组件

统一技术栈建议：

- 语言：Python 3.11+
- 编排：LangGraph（或 NexAU，如果你打算深挖 Nex 生态）
- API 层：FastAPI
- 数据层：SQLite（开发）/PostgreSQL（生产）
- 向量库：FAISS（本地）或 Milvus（可扩展）
- 文档解析：`pymupdf` + `unstructured` + `markdown-it-py`
- OCR（图片）：`rapidocr` 或 `paddleocr`
- 模型：DeepSeek / 硅基流动兼容 OpenAI 协议模型 / 本地 Qwen

---

## 4. 练习一：个人日程助手

## 4.1 需求拆解

必做能力：

1. 工具：新增日程、修改日程、查询日程、删除日程
2. 数据源：聊天记录、飞书文档、Markdown 笔记
3. Agent 能够从数据源中提取日程信息并落库
4. 支持自然语言交互（如“明天上午把 XX 改到 10 点”）

建议补充能力（提高可用性）：

1. 冲突检测（时间重叠）
2. 时区统一（默认 Asia/Shanghai）
3. 置信度与二次确认（低置信度操作先确认）
4. 操作审计日志

## 4.2 系统架构

```
[Client: Web/CLI]
      |
      v
[FastAPI Gateway]
      |
      v
[Agent Orchestrator(LangGraph/NexAU)]
      |---[Tool: schedule_create/update/delete/query]
      |---[Retriever: chat/feishu/md]
      |---[Parser: datetime/entity parser]
      |
      v
[Storage]
  - RDB: schedule/event/audit
  - VectorDB: note/chat embedding index
```

## 4.3 推荐目录结构

```text
agent-beginner/
  apps/
    scheduler_agent/
      api/
        main.py
        routes_chat.py
        routes_schedule.py
      agent/
        graph.py
        prompts.py
        planner.py
      tools/
        schedule_tools.py
        feishu_tools.py
        note_tools.py
      ingest/
        chat_loader.py
        feishu_loader.py
        markdown_loader.py
        chunker.py
      retrieval/
        embedder.py
        vector_store.py
        retriever.py
      core/
        config.py
        logger.py
        time_utils.py
      db/
        models.py
        repo.py
        migrations/
      tests/
        test_tools.py
        test_agent_flow.py
        test_api.py
  docs/
    TECHNICAL_IMPLEMENTATION_GUIDE.md
```

## 4.4 数据模型（最小可用）

### schedule_event

- `id` (uuid)
- `title` (str)
- `description` (text)
- `start_time` (datetime)
- `end_time` (datetime)
- `timezone` (str)
- `location` (str, nullable)
- `participants` (json)
- `source_type` (enum: chat/feishu/markdown/manual)
- `source_ref` (str)
- `confidence` (float)
- `status` (enum: confirmed/pending/cancelled)
- `created_at` / `updated_at`

### action_audit

- `id`
- `user_input`
- `agent_plan`
- `tool_name`
- `tool_args`
- `tool_result`
- `requires_confirmation` (bool)
- `created_at`

## 4.5 工具接口设计（核心）

统一工具协议（建议 JSON schema + Pydantic）：

### `schedule_create`
输入：

```json
{
  "title": "项目复盘",
  "start_time": "2026-04-10T10:00:00+08:00",
  "end_time": "2026-04-10T11:00:00+08:00",
  "timezone": "Asia/Shanghai",
  "description": "周会",
  "location": "线上"
}
```

输出：

```json
{
  "ok": true,
  "event_id": "uuid",
  "conflict": false,
  "message": "created"
}
```

### `schedule_update`

- 支持 `event_id` 精确更新
- 若无 `event_id`，先走 `schedule_query` 候选召回 + 让用户确认再改

### `schedule_query`

- 支持时间范围查询、关键词查询
- 输出候选事件数组，便于 Agent 后续 disambiguation

### `schedule_delete`

- 删除前必须二次确认

## 4.6 Agent 执行流程（建议状态机）

1. `IntentDetect`：识别是新增/修改/查询/删除/批量抽取
2. `ContextRetrieve`：从聊天记录 + 飞书 + Markdown 召回相关片段
3. `ExtractNormalize`：抽取时间、人物、地点、事件并标准化
4. `PlanAction`：决定调用哪个工具（及参数）
5. `SafetyCheck`：冲突检测、低置信度确认、危险操作确认
6. `ToolExecute`
7. `SummarizeResponse`：反馈操作结果
8. `AuditLog`

## 4.7 Prompt 策略

建议拆三段 Prompt：

1. `system_prompt`：工具使用规则、时间解析规则、禁止臆造 event_id
2. `planner_prompt`：把用户输入映射为标准 action
3. `extractor_prompt`：结构化抽取字段（JSON 输出）

关键约束：

- 缺少时间字段时不得直接创建日程，必须追问
- 解析“明天/下周一”时要基于当前时区和当前日期
- 修改/删除前输出候选项并确认

## 4.8 数据源接入实现

### 聊天记录

- 输入：本地 txt/json 或 IM 导出文件
- 实现：按会话分块 + 时间戳保留 + embedding 入库

### 飞书文档

- 方式 A：开放平台 API 拉取文档块
- 方式 B：先导出 Markdown，再走 Markdown 通道（低复杂度）

### Markdown

- 用标题层级切块（`#`, `##`, `###`）
- 每块保留来源文件与行号范围，便于回溯

## 4.9 API 设计（最小集合）

- `POST /chat`：用户输入，返回 agent 响应
- `POST /ingest/chat`
- `POST /ingest/feishu`
- `POST /ingest/markdown`
- `GET /schedule?from=&to=&q=`
- `PATCH /schedule/{event_id}`
- `DELETE /schedule/{event_id}`

## 4.10 验收标准（练习一）

1. 能从“聊天 + Markdown”中抽取至少 10 条日程并正确入库
2. 修改请求能正确定位目标事件（至少 90%）
3. 冲突检测可用（重叠时给出提醒）
4. 危险操作（删除）必须二次确认
5. 有完整 audit log

---

## 5. 练习二：NexDR 理解与改造

> 你需要先阅读 NexDR 源码，再按下述改造点落地。下面给的是“实现蓝图 + 修改策略”。

## 5.1 原系统（典型）流程抽象

通常会是：

1. 用户输入研究主题
2. 检索器执行网络搜索
3. Agent 汇总与写作
4. 产出 Markdown + HTML

## 5.2 改造目标 1：替换网络搜索为 Semantic Scholar

### 设计原则

- 抽象 `SearchProvider` 接口，避免把供应商逻辑写死
- 支持多后端：Semantic Scholar（主）+ arXiv/Crossref（备）

### 接口定义

```python
class SearchProvider(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[Paper]: ...
```

### `Paper` 结构

- `paper_id`
- `title`
- `abstract`
- `authors`
- `year`
- `venue`
- `url`
- `citation_count`

### 排序策略建议

综合得分：

`score = 0.45 * relevance + 0.25 * recency + 0.20 * citation + 0.10 * source_quality`

## 5.3 改造目标 2：不生成 HTML，改为“Markdown 人机协作迭代”

目标流程：

1. Agent 生成 `draft.md`
2. 用户手工修改 `draft.md`
3. Agent 识别用户改动（diff）
4. Agent 仅针对改动点和指令做增量修改
5. 输出 `draft_v2.md`（仍是 Markdown）

### 核心实现

- 引入 `RevisionEngine`
- 用 `difflib` 或 `git diff --no-index` 获取变更块
- 变更块转结构化对象：
  - `section`
  - `change_type` (add/delete/modify)
  - `before`
  - `after`
- Agent Prompt 中注入变更摘要，要求“最小化修改”

## 5.4 改造目标 3：支持 PDF/图片输入

### 输入管线

1. 文件识别：`pdf` / `image` / `md` / `txt`
2. 文本抽取：
   - PDF：`pymupdf`（优先）+ OCR 回退
   - 图片：OCR
3. 清洗与切块
4. 向量化入库
5. 检索增强生成

### 元数据要求

每个 chunk 存：

- `source_file`
- `page_no`（PDF）
- `bbox`（可选）
- `ocr_confidence`（图片）

## 5.5 建议新增模块

```text
nexdr_ext/
  providers/
    semantic_scholar.py
    arxiv_provider.py
    base.py
  revision/
    diff_parser.py
    revision_engine.py
  ingestion/
    pdf_ingestor.py
    image_ingestor.py
    multimodal_router.py
```

## 5.6 关键伪代码

### 1) 检索替换

```python
def retrieve_papers(query: str) -> list[Paper]:
    papers = semantic_provider.search(query, top_k=20)
    if len(papers) < 5:
        papers += arxiv_provider.search(query, top_k=10)
    return rerank(papers)
```

### 2) Markdown 迭代

```python
def revise_markdown(user_instruction, old_md, edited_md):
    changes = diff_parser.parse(old_md, edited_md)
    prompt = build_revision_prompt(user_instruction, changes, edited_md)
    new_md = llm.generate(prompt)
    return new_md
```

### 3) 多模态接入

```python
def ingest_file(path):
    t = detect_type(path)
    if t == "pdf":
        docs = pdf_ingestor.extract(path)
    elif t == "image":
        docs = image_ingestor.ocr(path)
    else:
        docs = text_ingestor.read(path)
    chunks = chunker.split(docs)
    vector_store.upsert(chunks)
```

## 5.7 验收标准（练习二）

1. 搜索调用已从网页搜索迁移到 Semantic Scholar
2. 输出仅 Markdown，无 HTML 产物
3. 用户改 `draft.md` 后，Agent 能识别 diff 并继续改写
4. PDF/图片输入可被成功解析并用于回答

---

## 6. 工程落地计划（两周版）

### 第 1-2 天：脚手架与基础能力

1. 初始化项目结构、配置管理、日志
2. 接通一个可用 LLM Provider
3. 建立 schedule 的 DB 模型与 CRUD

### 第 3-5 天：练习一核心闭环

1. 实现 `schedule_create/update/query/delete`
2. 实现 chat + markdown ingest 与检索
3. 完成 Agent 工作流（意图识别 -> 工具调用）
4. 增加冲突检测与二次确认

### 第 6-7 天：练习一测试与稳定性

1. 单测、集成测试
2. 修复时间解析与歧义匹配问题
3. 准备演示脚本

### 第 8-10 天：练习二检索改造

1. 抽象 SearchProvider
2. 接入 Semantic Scholar
3. 完成 rerank 与 citation 结构化

### 第 11-12 天：Markdown 迭代与多模态

1. 实现 diff parser + revision engine
2. 完成 PDF/图片 ingestion
3. 打通端到端流程

### 第 13-14 天：验收与答辩材料

1. 跑全量测试
2. 记录 benchmark 与失败案例
3. 整理思考题答案与架构图

---

## 7. 测试策略

## 7.1 单元测试

- 工具参数校验
- 时间解析（相对时间、跨天、跨时区）
- diff 解析准确性
- PDF/OCR 解析质量

## 7.2 集成测试

- 从输入一句自然语言到日程落库的全链路
- 用户编辑 Markdown 后二次改写链路
- PDF+图片混合输入链路

## 7.3 回归测试集（建议）

至少准备：

1. 50 条日程语句（新增/修改/删除/歧义）
2. 20 组论文检索 query
3. 10 份 PDF + 20 张图片 OCR 样本

---

## 8. 思考题的工程化回答

## 8.1 MCP 与 Skills 常见用法

1. MCP：将外部能力（数据库、文档系统、知识库、企业工具）标准化成可调用上下文源
2. Skills：把“某类任务的固定工作流”沉淀为可复用指令包（例如论文检索 Skill、报告修订 Skill）
3. 最佳实践：
   - 一个 Skill 只做一类任务
   - 明确输入输出 schema
   - 加入失败回退策略（例如 provider 切换）

## 8.2 提高 Agent 部署时显卡利用率

1. 批处理请求（batch decode）
2. 开启 KV Cache 与连续批处理
3. 长上下文采用分段检索，减少无效 token
4. 模型量化（AWQ/GPTQ）+ 张量并行
5. 推理框架选型（vLLM/TGI）并调整 `max_num_seqs`、`gpu_memory_utilization`

## 8.3 利用环境反馈训练更好的 Agent

1. 记录轨迹：用户输入、工具调用序列、结果质量
2. 建立反馈标签：成功/失败、人工修订点、耗时
3. 训练路径：
   - 监督微调：学习高质量工具调用轨迹
   - 偏好优化：用人类偏好排序优化响应
   - 离线强化学习：基于任务完成率和代价函数优化策略

---

## 9. 最小可行实现（MVP）定义

若你时间有限，先做以下最小闭环：

1. 仅接 Markdown + 聊天记录（先不接飞书 API）
2. 仅支持新增/查询/修改三类日程操作
3. 练习二先完成“Semantic Scholar 替换 + Markdown 迭代”，最后再补 PDF/图片

这样可以先交付可演示版本，再逐步扩展。

---

## 10. 你可以直接开始的第一步

1. 按第 4.3 节创建目录结构
2. 先完成 `schedule_event` + 4 个工具函数
3. 接一条最简单 Agent 流程：用户输入 -> 调用工具 -> 返回结果
4. 加入 Markdown ingest，做第一版“从笔记抽取日程”

完成这 4 步后，就已经有可运行雏形。
