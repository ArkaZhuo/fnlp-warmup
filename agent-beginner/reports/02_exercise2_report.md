# 练习二报告：NexDR 的理解与修改

## 1. README 要求里“要做什么”
根据仓库 `README.md`，练习二需要完成：

1. 阅读并理解 NexDR 代码。
2. 使用其它工具替代网络搜索（示例：Semantic Scholar API）。
3. 不生成 HTML；在 markdown 生成后允许用户修改，Agent 识别修改并继续改。
4. 支持更多输入类型，包括 PDF、图片等。

## 2. 本项目“怎么做”

### 2.1 搜索替换
- 文件：`exercise2_nexdr/search/semantic_scholar.py`
- 路由：`exercise2_nexdr/search/search_router.py`
- 关键点：
  - 接入 Semantic Scholar Graph API
  - 使用加权重排：`0.45 relevance + 0.25 recency + 0.20 citation + 0.10 source_quality`
  - 对 `429/5xx` 做容错，避免流程中断

### 2.2 Markdown-only 与用户改稿迭代
- 入口：`exercise2_nexdr/run_exercise2.py`
- 差异识别：`exercise2_nexdr/revision/diff_parser.py`
- 迭代修订：`exercise2_nexdr/revision/revision_engine.py`
- 输出：
  - 初稿：`markdown_report_v1.md`
  - 迭代稿：`markdown_report_v2.md`
  - 元数据：`run_metadata.json`

### 2.3 多模态输入
- 文件：`exercise2_nexdr/ingestion/multimodal_ingestor.py`
- 支持类型：`txt/md/pdf/image`
- 处理策略：
  - PDF：文本抽取优先，空页 fallback OCR/vision
  - 图片：优先 tesseract，失败后可走 `MULTI_MODAL_LLM_*`

## 3. 整理后的结果（含图片）

### 3.1 结果图
![练习二流水线结果](./assets/exercise2_pipeline_result.png)

### 3.2 图中结果说明
本次展示了两个工作空间结果：

- `demo_v1_nlp`：
  - `paper_count=0`
  - `input_file_count=3`
  - `ingested_chunk_count=3`
  - `retrieved_chunk_count=3`
  - 输出仅含 `markdown_report_v1`
- `demo_v2_nlp`（带用户编辑迭代）：
  - `paper_count=1`
  - `input_file_count=1`
  - `ingested_chunk_count=1`
  - `retrieved_chunk_count=1`
  - 输出包含 `markdown_report_v1 + markdown_report_v2`
- 两次运行 `html_generation` 均为 `false`，符合“禁用 HTML”要求。

### 3.3 验证结果
- 测试目录：`exercise2_nexdr/tests/`
- 复跑结果：`Ran 6 tests in 0.586s, OK`
- 覆盖了：
  - Semantic Scholar 重排
  - Markdown diff 识别
  - 多模态 ingestion 基础能力

## 4. 与 README 的对应结论
- 替代搜索：已完成（Semantic Scholar）。
- markdown-only + 用户改稿迭代：已完成（v1/v2 管线）。
- PDF/图片输入：已完成（多模态 ingestion）。

## 5. 我的思考
1. 练习二的实质是“把研究流程从单轮生成改造成可迭代工程流程”，比单次生成更接近真实场景。
2. 搜索层要把“可用性”放在第一位，429/网络异常时不中断是必要工程约束。
3. 多模态能力上，OCR 质量高度依赖外部能力；建议在答辩时明确“基础可用 + 可插拔增强”的设计思路。
