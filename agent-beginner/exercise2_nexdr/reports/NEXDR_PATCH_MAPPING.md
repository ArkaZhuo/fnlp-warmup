# NexDR 改造映射说明

本文件说明 `exercise2_nexdr` 与原 NexDR 模块的对应关系，便于将实现迁移回 NexDR 主仓。

## 1. 搜索替换（Semantic Scholar）

原始模块：
- `nexdr/agents/deep_research/search.py`
- `nexdr/agents/deep_research/web_search.py`
- `nexdr/agents/deep_research/serper_search.py`

改造实现：
- `exercise2_nexdr/search/semantic_scholar.py`
- `exercise2_nexdr/search/search_router.py`

迁移建议：
1. 在原 `search.py` 中将 `search_source=web` 分支改为调用 Semantic Scholar provider。
2. 保留 arxiv 分支可选；主分支默认 semantic_scholar。

## 2. Markdown-only 与用户编辑迭代

原始模块：
- `quick_start.py`（支持 markdown/html/markdown+html）

改造实现：
- `exercise2_nexdr/run_exercise2.py`
- `exercise2_nexdr/revision/diff_parser.py`
- `exercise2_nexdr/revision/revision_engine.py`

迁移建议：
1. 在 `quick_start.py` 中移除 html 分支或保留但默认禁用。
2. 新增“用户编辑稿输入”参数，触发 markdown v2 修订流程。

## 3. PDF/图片输入支持

原始模块：
- `nexdr/agents/doc_reader/file_parser.py`
- `nexdr/agents/doc_reader/doc_preprocess.py`

改造实现：
- `exercise2_nexdr/ingestion/multimodal_ingestor.py`
- `exercise2_nexdr/ingestion/retrieval.py`

迁移建议：
1. 在 file_parser 里引入 pdf/image 路由。
2. PDF 优先文本抽取，空页 fallback OCR/vision。
3. 图片优先 OCR，失败 fallback 多模态模型。
