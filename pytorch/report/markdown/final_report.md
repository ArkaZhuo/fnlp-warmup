# Final Report（Task-1/2/3 总结）

## 1. 项目目标与范围
- 本报告按文档要求，仅覆盖 Task-1、Task-2、Task-3。
- 数据与实验结果来源于 `outputs/` 下已完成运行。
- 详细过程见：`task1_report.md`、`task2_report.md`、`task3_report.md`。

## 2. 与 README 要求对齐情况
- Task-1：完成 BoW/N-gram、loss/lr 对照与图表。
- Task-2：完成 loss/lr、卷积核/优化器、GloVe、CNN/RNN/Transformer 对照与图表。
- Task-3：完成加法子任务、不同划分泛化、decoder-only 变种、语言模型子任务与图表。

## 3. 关键结果总表
| Task | Best Run | Core Metric | Value | 关键配置 |
|---|---|---:|---:|---|
| Task-1 | `best_ngram2_ce_lr01_30k` | test_acc | **0.3714** | mode=ngram, loss=ce, lr=0.1 |
| Task-2 | `best_rnn_e8` | test_acc | **0.4859** | model=rnn, loss=ce, opt=adam, lr=0.001 |
| Task-3 子任务1 | `add_easy_rev_big` | exact_match | **0.1483** | d_model=256, nhead=8, reverse=(True,True) |
| Task-3 子任务2 | `req_lm_d128_seq64` | val_ppl (lower better) | **10.1624** | d_model=128, seq_len=64 |

### 3.1 GloVe 对照（Task-2 重点）
- 随机初始化（50d）：test_acc = 0.4056
- GloVe 初始化（50d）：test_acc = **0.4811**（loaded vectors = 7599）

## 4. 图表总览
### Task-1
Figure 1：Task1 不同实验配置的测试集准确率对比图。横坐标表示实验名称，例如 `exp_06_ngram_ce_lr0.2` 表示“第 6 组实验 + N-gram 特征 + 交叉熵损失 + 学习率 0.2”；纵坐标表示测试集准确率 `test_acc`，柱子越高表示该实验配置越有效。
![task1_summary](/home/df_05/A_fnlp/pytorch/outputs/task1/final/experiment_summary.png)
Figure 2：Task1 最优模型训练曲线图。横坐标表示训练轮数 `epoch`；纵坐标分别表示损失 `loss` 和准确率 `accuracy` 的变化，用来观察模型收敛速度与后期是否进入平台期。
![task1_best_curve](/home/df_05/A_fnlp/pytorch/outputs/task1/best/best_ngram2_ce_lr01_30k/training_curve.png)

### Task-2
Figure 3：Task2 不同实验配置的测试集准确率对比图。横坐标表示实验名称，如 `exp_03_rnn_ce_adam_lr0.001` 依次对应实验编号、模型类型、损失函数、优化器和学习率；纵坐标表示测试集准确率 `test_acc`，用于比较 CNN、RNN、Transformer 及不同超参数的效果。
![task2_summary](/home/df_05/A_fnlp/pytorch/outputs/task2/final/experiment_summary.png)
Figure 4：Task2 最优模型训练曲线图。横坐标表示训练轮数 `epoch`；纵坐标表示损失 `loss` 和准确率 `accuracy` 的变化，可用于判断模型是否稳定收敛以及是否出现过拟合。
![task2_best_curve](/home/df_05/A_fnlp/pytorch/outputs/task2/best/best_rnn_e8/training_curve.png)

### Task-3
Figure 5：Task3 加法任务最优模型的训练曲线图。横坐标表示训练轮数 `epoch`；纵坐标同时展示训练损失 `train_loss` 和测试集精确匹配率 `test_exact`，用于观察模型是否真正学会完整输出正确答案。
![task3_add_best_curve](/home/df_05/A_fnlp/pytorch/outputs/task3/best/add_easy_rev_big/curve.png)
Figure 6：Task3 语言模型最优实验的训练曲线图。横坐标表示训练轮数 `epoch`；纵坐标表示训练损失 `train_loss` 和验证损失 `val_loss`，反映语言模型在字符级预测上的拟合与泛化情况。
![task3_lm_best_curve](/home/df_05/A_fnlp/pytorch/outputs/task3/req/req_lm_d128_seq64/curve.png)

### Task-3 语言模型样例（截断）
> this and the the the the the the the the sthe the the the the the the the sthe the the the the the sthe the the the the the t

## 5. 结论与思考
1. 从 Task-1 到 Task-2，模型能力提升明显，说明序列建模和预训练信息对该情感分类数据更关键。
2. Task-2 中 GloVe 初始化带来显著提升，验证了外部语义先验对分类性能有直接收益。
3. Task-3 的加法任务在 easy 划分表现明显好于 base/iid，泛化短板主要体现在分布外组合。
4. Task-3 语言模型困惑度已稳定下降，但样例文本仍偏重复，后续可通过更长训练、更大模型或更优 tokenizer 继续改进。

## 6. 附录：分任务详细报告
- [Task1 Report](task1_report.md)
- [Task2 Report](task2_report.md)
- [Task3 Report](task3_report.md)
