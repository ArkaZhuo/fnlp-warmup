# agent-beginner

这个目录包含两个练习：

- `exercise1_nexau/`：基于 NexAU 的个人日程助手
- `exercise2_nexdr/`：对 NexDR 的搜索、修订和多模态输入流程改造

这里只保留核心代码运行说明，不包含报告内容。

## 目录结构

```text
exercise1_nexau/       练习一：个人日程助手
exercise2_nexdr/       练习二：NexDR 改造
third_party/           上游依赖源码，仅供参考
```

## 环境准备

建议先进入目录：

```bash
cd /home/df_05/A_fnlp/agent-beginner
```

建议使用当前 `nlp` 环境：

```bash
conda activate nlp
```

如果是练习一，需要本地可用的 NexAU 安装，以及可选的 `.env` / API 环境变量。

## 练习一：exercise1_nexau

查看子目录完整说明：

```bash
sed -n '1,260p' exercise1_nexau/README.md
```

### 1. 终端版后端 Demo

```bash
cd /home/df_05/A_fnlp/agent-beginner
/home/df_05/anaconda3/envs/nlp/bin/python -m exercise1_nexau.run_terminal_demo
```

### 2. Agent 交互入口

```bash
cd /home/df_05/A_fnlp/agent-beginner
/home/df_05/anaconda3/envs/nlp/bin/python -m exercise1_nexau.run_scheduler_agent
```

单轮消息测试：

```bash
cd /home/df_05/A_fnlp/agent-beginner
/home/df_05/anaconda3/envs/nlp/bin/python -m exercise1_nexau.run_scheduler_agent \
  --message "把明天 10:00 的项目复盘加到日程"
```

### 3. 网页演示

```bash
cd /home/df_05/A_fnlp/agent-beginner
/home/df_05/anaconda3/envs/nlp/bin/python -m exercise1_nexau.demo_web --host 127.0.0.1 --port 8008
```

浏览器打开：

```text
http://127.0.0.1:8008
```

### 4. 单元测试

```bash
cd /home/df_05/A_fnlp/agent-beginner
/home/df_05/anaconda3/envs/nlp/bin/python -m unittest exercise1_nexau.tests.test_scheduler_tools
```

## 练习二：exercise2_nexdr

查看子目录完整说明：

```bash
sed -n '1,260p' exercise2_nexdr/README.md
```

### 1. 生成第一版 Markdown 报告

```bash
cd /home/df_05/A_fnlp/agent-beginner
/home/df_05/anaconda3/envs/nlp/bin/python -m exercise2_nexdr.run_exercise2 \
  --query "How to improve RAG factuality in production?" \
  --inputs "exercise2_nexdr/samples/sample_notes.md,exercise2_nexdr/samples/sample_pdf.pdf,exercise2_nexdr/samples/sample_image.png" \
  --output_dir "exercise2_nexdr/workspaces/demo_v1"
```

### 2. 基于用户修改稿继续修订

```bash
cd /home/df_05/A_fnlp/agent-beginner
/home/df_05/anaconda3/envs/nlp/bin/python -m exercise2_nexdr.run_exercise2 \
  --query "How to improve RAG factuality in production?" \
  --inputs "exercise2_nexdr/samples/sample_notes.md" \
  --edited_markdown "exercise2_nexdr/samples/user_edited_markdown.md" \
  --user_instruction "Add a practical deployment checklist and risk controls section." \
  --output_dir "exercise2_nexdr/workspaces/demo_v2"
```

### 3. 单元测试

```bash
cd /home/df_05/A_fnlp/agent-beginner
/home/df_05/anaconda3/envs/nlp/bin/python -m unittest exercise2_nexdr.tests.test_diff_parser
/home/df_05/anaconda3/envs/nlp/bin/python -m unittest exercise2_nexdr.tests.test_multimodal_ingestor
/home/df_05/anaconda3/envs/nlp/bin/python -m unittest exercise2_nexdr.tests.test_semantic_scholar_rank
```
