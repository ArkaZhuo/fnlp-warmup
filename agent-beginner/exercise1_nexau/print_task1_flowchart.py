from __future__ import annotations


FLOWCHART = r"""
========================================================================
  Task1 Flowchart
========================================================================

  [命令行输入]
  聊天指令 / 飞书链接 / Markdown 路径
          |
          v
  [终端 Demo / 调度入口]
  run_terminal_demo.py / run_scheduler_agent.py
          |
          v
  [NexAU 框架 Agent]
          |
          +---------------------> [Model API]
          |                       DeepSeek / OpenAI 兼容接口
          |
          v
  [本地工具层]
  创建 / 查询 / 修改 / 删除 / 导入
      |                     |
      v                     v
  [SQLite 日程库]      [飞书开放平台 API]
      |                     |
      |                     v
      |               [飞书文档 / 飞书日历]
      |
      v
  [命令行结果输出]

========================================================================
"""


def main() -> None:
    print(FLOWCHART)


if __name__ == "__main__":
    main()
