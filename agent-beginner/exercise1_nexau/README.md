# Exercise 1: NexAU 个人日程助手（可运行实现）

这个目录是练习一的落地代码，使用了 **NexAU 框架** 来构建工具调用型 Agent。

## 功能覆盖

- 日程工具：`create / query / update / delete`
- 冲突检测：时间冲突时阻断并提示
- 删除二次确认：`schedule_delete(confirm=true)` 才会取消日程
- 文档抽取：支持从以下文件中抽取时间行并创建/更新日程
  - Markdown 笔记
  - 聊天记录 txt
  - 飞书导出的 Markdown
  - 飞书开放平台 Docx API（直连）

## 目录说明

- `run_scheduler_agent.py`：NexAU Agent 启动入口（交互式/单次）
- `scheduler_tools.py`：自定义工具与 SQLite 存储实现
- `tools/*.tool.yaml`：NexAU 工具 schema
- `prompts/system_prompt.md`：系统提示词
- `data/`：示例输入文件与默认数据库路径

## 环境准备

1. 安装 NexAU（任选其一）

```bash
# 方式 A：你本地已有 NexAU 源码
pip install -e /path/to/NexAU

# 方式 B：通过 GitHub（需要权限）
pip install "git+ssh://git@github.com/nex-agi/NexAU.git@v0.4.1"
```

2. 可选：安装 dotenv

```bash
pip install python-dotenv
```

3. 配置环境变量（OpenAI 兼容接口）

```bash
export LLM_MODEL="deepseek-chat"
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_API_KEY="your_api_key"
# 可选
export LLM_API_TYPE="openai_chat_completion"
export SCHEDULE_TIMEZONE="Asia/Shanghai"
export SCHEDULE_DB_PATH="./exercise1_nexau/data/schedule.db"
# 飞书直连（可选）
export FEISHU_APP_ID="cli_xxx"
export FEISHU_APP_SECRET="xxx"
export FEISHU_BASE_URL="https://open.feishu.cn"
```

## 运行方式

### 交互模式

```bash
cd exercise1_nexau
python run_scheduler_agent.py
```

### 单次执行

```bash
cd exercise1_nexau
python run_scheduler_agent.py --message "把明天 10:00 的项目复盘加到日程"
```

## 推荐演示脚本

1. 让 Agent 先导入 markdown 笔记：

```text
请从 ./data/sample_notes.md 同步日程，使用 upsert
```

2. 查询未来 7 天日程：

```text
帮我查询未来7天的日程
```

3. 修改一个事件：

```text
把“项目复盘会议”改到后天上午11点
```

4. 删除事件（观察二次确认流程）：

```text
删除团队聚餐
```

5. 飞书直连导入（需要已配置 FEISHU_APP_ID/FEISHU_APP_SECRET）：

```text
请从飞书文档 https://xxx.feishu.cn/docx/doxcnxxxxxx 同步日程，使用 upsert
```

也支持传 wiki 链接：

```text
请从飞书文档 https://xxx.feishu.cn/wiki/xxxxxxxx 同步日程，使用 upsert
```

## 网页界面演示

如果需要录屏展示“从飞书文档同步日程、创建日程、修改日程、查询日程”，可以启动本地网页演示：

```bash
cd /home/df_05/A_fnlp/agent-beginner
/home/df_05/anaconda3/envs/nlp/bin/python -m exercise1_nexau.demo_web --host 127.0.0.1 --port 8008
```

浏览器打开：

```text
http://127.0.0.1:8008
```

页面中可以粘贴飞书 `docx/wiki` 链接并点击“同步飞书文档（upsert）”，也可以直接创建日程、保存修改、取消日程，或点击“导入示例 Markdown”展示从笔记中同步日程。

如果要使用飞书直连同步，启动前需要带上飞书应用环境变量，例如：

```bash
cd /home/df_05/A_fnlp/agent-beginner
FEISHU_APP_ID="cli_xxx" \
FEISHU_APP_SECRET="xxx" \
FEISHU_BASE_URL="https://open.feishu.cn" \
FEISHU_SYNC_CALENDAR="true" \
FEISHU_CALENDAR_ID="primary" \
/home/df_05/anaconda3/envs/nlp/bin/python -m exercise1_nexau.demo_web --host 127.0.0.1 --port 8008
```

说明：

- `FEISHU_SYNC_CALENDAR=true` 表示在同步到本地 SQLite 的同时，也尝试写入飞书真实日历。
- `FEISHU_CALENDAR_ID=primary` 表示默认写入主日历；如果你有其他可写日历，也可以改成对应的 calendar_id。
- 需要在飞书开放平台额外开启日历读写权限，否则只能读文档，不能创建真实日历事件。

## 说明

- 已支持两种 Feishu 接入方式：导出 Markdown 导入、以及飞书开放平台 Docx API 直连导入。
- 飞书直连导入默认从环境变量读取 `FEISHU_APP_ID` 与 `FEISHU_APP_SECRET`。
