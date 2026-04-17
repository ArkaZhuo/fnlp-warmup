# Exercise 1: Personal Schedule Agent

这个目录实现了一个“小而完整”的个人助理 Agent，用于演示：

1. 用户自然语言请求理解
2. 聊天记录 / Markdown 笔记 / 飞书导出文本检索
3. 结构化事件信息抽取
4. 缺参追问
5. 日程新增、查询、修改

## 目录结构

- `agent.py`: 主要 Agent 逻辑
- `run_demo.py`: 命令行交互入口
- `web_app.py`: 简易网页聊天界面
- `tools.py`: 日程工具
- `retrievers.py`: 多源文本检索
- `schemas.py`: 数据结构
- `data/chat_history/`: 聊天记录样例
- `data/notes/`: Markdown 笔记样例
- `data/feishu/`: 飞书导出文本样例
- `results/calendar_events.json`: 日程持久化文件

## 快速运行

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 agent_exercise1/run_demo.py
```

网页聊天界面：

```bash
/home/df_05/anaconda3/envs/nlp/bin/python3 agent_exercise1/web_app.py
```

然后打开：

```text
http://127.0.0.1:8008
```

## 示例输入

```text
帮我把明天下午和李老师讨论论文的会加到日程里，大概两点，地点可能在光华楼，顺便看看我之前笔记里有没有提过具体会议室。
```

```text
我明天下午有什么安排？
```

```text
把和导师的会延后一小时
```

## 说明

- 当前实现不直接连接真实飞书 API，而是通过 `data/feishu/` 中的导出文本模拟“飞书文档检索”。
- 检索采用轻量关键词召回 + 打分，不依赖外部模型或网络服务。
- 新版本支持多轮缺参追问，例如先提出“帮我安排和导师的会”，再逐轮补日期和时间。
- Agent 输出中会包含：
  - 意图判断
  - 检索命中
  - 工具选择
  - 结构化参数
  - 执行结果
