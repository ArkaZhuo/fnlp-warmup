# Final Report

## 说明
本目录下的 Markdown 报告按照 `README.md` 的要求组织为六份分任务报告，并补充一个总入口文件。

## 报告列表
- [Task1 Report](task1_report.md)
- [Task2 Report](task2_report.md)
- [Task3 Report](task3_report.md)
- [Task4 Report](task4_report.md)
- [Task5 Report](task5_report.md)
- [Task6 Report](task6_report.md)

## 整体说明
这些报告并非函数级别的代码注释，而是围绕每个任务回答以下四个问题：
- 这个任务在 README 里要求做什么。
- 我实际是怎样把它实现出来的。
- 图里呈现了什么现象，结果说明了什么。
- 从实验结果出发，我对这个任务有什么进一步思考。

## 简要总结
Task1 体现的是传统机器人学路线：通过逆运动学和轨迹设计，稳定完成抓取与放置，其优点在于过程可解释、误差可控。  
Task2 体现的是强化学习路线：先在离散 Gym 环境中验证算法流程，再迁移到机械臂抓取环境，核心在于将任务抽象为可学习的状态和动作结构。  
Task3 体现的是模仿学习路线：通过专家演示生成离线数据，再分别训练 BC 和 Diffusion 风格策略，以观察不同模仿方法在当前任务复杂度下的表现。  
Task4 体现的是 VLA 路线：通过图像、语言和动作三元组构造一个最小多模态训练闭环，并验证视觉语言到动作预测的可行性。  
Task5 体现的是 LLM/VLM 规划路线：在统一 benchmark 上比较 zero-shot、ICL、CoT 与 SFT 四类规划范式，分析其在不同复杂度任务上的差异。  
Task6 体现的是人形机器人控制路线：以全身遥操作轨迹为基础，通过模仿学习和强化学习残差微调逐步逼近参考控制器。
