# 练习二技术报告（NexDR 理解与改造）

## 1. 目标
基于课程要求，对 NexDR 工作流完成三项改造：

1. 将通用 web 搜索替换为 Semantic Scholar API
2. 取消 HTML 生成，改为 markdown 人机协作迭代
3. 增强输入通道，支持 PDF、图片等多模态输入

## 2. 原始 NexDR 关键路径梳理

参考代码（来自 NexDR 仓库）：

- 搜索入口：`nexdr/agents/deep_research/search.py`
- Web 搜索实现：`nexdr/agents/deep_research/web_search.py`
- Serper 细节：`nexdr/agents/deep_research/serper_search.py`
- 输出控制：`quick_start.py`（`markdown/html/markdown+html`）
- 文档读取入口：`nexdr/agents/doc_reader/doc_preprocess.py`
- 文件解析：`nexdr/agents/doc_reader/file_parser.py`

观察结论：

1. 原搜索以 `web_search -> Serper` 为主
2. 原运行器显式支持 HTML 输出
3. 原文件读取对本地二进制（PDF/图片）解析能力有限，图片主要依赖独立工具

## 3. 改造实现概览

改造代码位于 `exercise2_nexdr/`。

### 3.1 搜索替换：Semantic Scholar

- 文件：`search/semantic_scholar.py`
- 文件：`search/search_router.py`

实现点：

1. 接入 `https://api.semanticscholar.org/graph/v1/paper/search`
2. 字段抽取：`paperId/title/abstract/year/venue/url/citationCount/authors`
3. 增加重排评分：
   - `0.45 * relevance`
   - `0.25 * recency`
   - `0.20 * citation`
   - `0.10 * source_quality`
4. 保留兼容入口：若旧路径传 `search_source=web`，内部重定向到 Semantic Scholar
5. 增加限流容错：Semantic Scholar 返回 429/5xx 时不中断主流程，继续生成 markdown

### 3.2 输出改造：Markdown-only + 用户改稿迭代

- 文件：`core/report_builder.py`
- 文件：`revision/diff_parser.py`
- 文件：`revision/revision_engine.py`
- 文件：`run_exercise2.py`

实现点：

1. 主流程只生成 `markdown_report_v1.md`，不生成 HTML
2. 用户可手工编辑 markdown 后，输入：
   - `old markdown`
   - `edited markdown`
   - `user_instruction`
3. 系统识别修改内容：
   - section 级别 add/delete/modify
   - unified diff
4. 在用户编辑基础上生成 `markdown_report_v2.md`（增量修订）

### 3.3 多模态输入：PDF、图片、文本

- 文件：`ingestion/multimodal_ingestor.py`
- 文件：`ingestion/retrieval.py`

实现点：

1. 文本输入：`txt/md/...` 直接读取并切块
2. PDF 输入：
   - 优先 `PyMuPDF` 文本抽取
   - 页面无文本时：渲染图片 + OCR/多模态 fallback
3. 图片输入：
   - 优先 `pytesseract`（若环境可用）
   - 否则用多模态 LLM（若配置 `MULTI_MODAL_LLM_*`）
   - 再否则输出可解释 fallback 提示
4. 检索：基于轻量词项相似度从 chunk 中召回证据片段

## 4. 运行与产物

运行入口：`run_exercise2.py`

典型产物：

- `markdown_report_v1.md`
- `markdown_report_v2.md`（可选）
- `run_metadata.json`

## 5. 测试与验证

测试文件：

- `tests/test_diff_parser.py`
- `tests/test_semantic_scholar_rank.py`
- `tests/test_multimodal_ingestor.py`

测试覆盖：

1. Markdown 变更识别
2. Semantic Scholar 重排逻辑
3. Markdown/PDF/图片 ingestion 基础可用性

## 6. 对课程要求的逐项对应

1. 使用其它工具替代网络搜索：已完成（Semantic Scholar API）
2. 不生成 HTML，支持用户改 markdown 后继续修改：已完成（v1/v2 流程）
3. 支持 PDF、图片输入：已完成（多模态 ingestion）

## 7. 后续可选增强

1. 将当前实现直接打补丁回 NexDR 原仓库（替换对应模块）
2. 增加向量检索后端（FAISS/Milvus）
3. 为图片 OCR 增加专用引擎（PaddleOCR/RapidOCR）提高中文识别率
