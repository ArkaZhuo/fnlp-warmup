练习一：智能体体验与原理初探
- 使用 NexN1:free 的 API （目前已不免费）、硅基流动的 API 或 DeepSeek 的 API 或者本地部署的 Qwen 系列模型搭建智能体
- 搭建目标：个人日程助手。
  - 自己设计几个添加日程、修改日程的工具；
  - 智能体使用这些工具可以从你的聊天记录和飞书文档/markdown 笔记中创建、修改日程。
- 可以使用 https://github.com/nex-agi/NexAU 框架，也可以考虑使用其他框架
- （可选）深入阅读 NexAU 的代码实现


练习二：NexDR 的理解与修改
阅读理解 https://github.com/nex-agi/NexDR 的代码
进行以下修改
- 使用其它工具替代网络搜索，例如（Semantic Scholar API ）
- 不生成 html；在生成完 markdown 后可以由用户修改，Agent 识别用户修改的内容，再进行进一步的修改
- 支持更多内容的输入，包含 PDF，图片等


思考题
1. 熟悉 MCP 和 Skills ，总结常见的使用方法
2. 如何提高 Agent 部署时显卡使用率及运行
3. 如何利用环境的反馈训练更好的 Agent
