# 练习一报告：智能体体验与原理初探（NexAU 个人日程助手）

## 1. README 要求里“要做什么”
根据仓库 `README.md`，练习一目标是：

1. 使用可用 LLM API（DeepSeek / 硅基流动 / 本地 Qwen）搭建智能体。
2. 搭建“个人日程助手”。
3. 自己设计添加、修改日程工具。
4. 智能体可从聊天记录和飞书文档/Markdown 笔记中创建、修改日程。
5. 可使用 NexAU 框架实现。

## 2. 本项目“怎么做”

### 2.1 架构与入口
- 运行入口：`exercise1_nexau/run_scheduler_agent.py`
- 工具实现：`exercise1_nexau/scheduler_tools.py`
- 工具协议：`exercise1_nexau/tools/*.tool.yaml`
- 提示词：`exercise1_nexau/prompts/system_prompt.md`

### 2.2 已实现能力
- 日程工具：`schedule_create / schedule_query / schedule_update / schedule_delete`
- 冲突检测：创建或更新时检查时间重叠
- 删除确认：`schedule_delete(confirm=true)` 二次确认
- 多源导入：
  - 聊天记录（txt）
  - Markdown 笔记
  - 飞书导出的 Markdown
  - 飞书开放平台 Docx API（支持 docx URL / wiki URL / token）
- 存储：SQLite（`exercise1_nexau/data/schedule.db`）

## 3. 整理后的结果（含图片）

### 3.1 结果图
![练习一结果总览](./assets/exercise1_result_overview.png)

### 3.2 图中结果说明
- 当前库内事件总数：`8`
- 来源分布：`feishu_doc=4`，`markdown=4`
- 状态分布：`confirmed=7`，`cancelled=1`
- 时间覆盖范围（UTC）：`2026-04-10T01:00:00+00:00` 到 `2026-04-13T07:00:00+00:00`

### 3.3 验证结果
- 单元测试：`exercise1_nexau/tests/test_scheduler_tools.py`
- 复跑结果：`Ran 4 tests in 0.172s, OK`
- 实际运行中已完成飞书同步 upsert，并出现“更新已有事件”的正确行为（updated_count>0）。

## 4. 与 README 的对应结论
- README 要求的核心能力（工具设计 + 多源导入 + 日程创建/修改）已覆盖。
- NexAU 框架已实际用于 agent 构建与工具调用链路。
- 练习一可以按“可运行、可验证、可演示”标准提交。

## 5. 我的思考
1. 练习一最关键不是“能调 API”，而是把工具边界设计清楚（尤其是修改/删除前确认）。
2. 飞书集成里最容易出问题的是权限与文档级授权，工程上应把权限检测前置为启动自检。
3. 当前实现已经可交付；若继续提升，优先做“时间表达鲁棒性（相对时间、多语言格式）”和“可观测性日志”。
