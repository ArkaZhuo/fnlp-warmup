# Exercise 2: NexDR 改造实现（可运行）

本目录实现了 README 中练习二的三个要求：

1. 用 Semantic Scholar API 替代通用网页搜索
2. 不生成 HTML，只走 Markdown；支持“用户改稿后再迭代”
3. 支持更多输入：PDF、图片、Markdown/TXT

## 代码结构

- `search/semantic_scholar.py`：Semantic Scholar 检索与重排
- `search/search_router.py`：搜索路由（兼容旧 `web` 调用路径，内部转 S2）
- `revision/diff_parser.py`：识别用户修改（section diff + unified diff）
- `revision/revision_engine.py`：基于用户编辑稿做二次修订
- `ingestion/multimodal_ingestor.py`：PDF/图片/文本 ingestion
- `run_exercise2.py`：端到端运行入口（markdown-only）

## 环境变量

最少需要：

```bash
export LLM_MODEL="deepseek-chat"
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_API_KEY="<your_key>"
```

可选：

```bash
# Semantic Scholar API key（有则更稳）
export SEMANTIC_SCHOLAR_API_KEY="<your_s2_key>"

# 图片 OCR fallback 到多模态模型时使用
export MULTI_MODAL_LLM_MODEL="<vision_model>"
export MULTI_MODAL_LLM_BASE_URL="<vision_base_url>"
export MULTI_MODAL_LLM_API_KEY="<vision_api_key>"
```

## 运行示例

### 1) 生成 markdown v1（不生成 html）

```bash
python -m exercise2_nexdr.run_exercise2 \
  --query "How to improve RAG factuality in production?" \
  --inputs "exercise2_nexdr/samples/sample_notes.md,exercise2_nexdr/samples/sample_pdf.pdf,exercise2_nexdr/samples/sample_image.png" \
  --output_dir "exercise2_nexdr/workspaces/demo_v1"
```

### 2) 用户编辑 markdown 后做迭代修订（输出 v2）

```bash
python -m exercise2_nexdr.run_exercise2 \
  --query "How to improve RAG factuality in production?" \
  --inputs "exercise2_nexdr/samples/sample_notes.md" \
  --edited_markdown "exercise2_nexdr/samples/user_edited_markdown.md" \
  --user_instruction "Add a practical deployment checklist and risk controls section." \
  --output_dir "exercise2_nexdr/workspaces/demo_v2"
```

## 输出文件

- `markdown_report_v1.md`
- `markdown_report_v2.md`（仅在传入 edited_markdown + user_instruction 时生成）
- `run_metadata.json`

说明：本实现明确禁用 HTML 输出。
