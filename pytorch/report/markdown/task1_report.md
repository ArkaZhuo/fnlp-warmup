# Task-1 Markdown Report（按 README 要求）

## 代码实现阶段
- 使用 `new_train.tsv / new_test.tsv` 读取数据（`src/task1/data.py`）。
- 使用 `train_test_split` 从训练集切分验证集。
- 使用 `NgramVectorizer` 实现 `BoW / N-gram` 特征。
- 使用纯张量线性分类器（不调用 `torch.nn`）进行 mini-batch 训练。
- 在训练过程中监控 `loss` 与验证集准确率，最终在测试集评估。

## 实验阶段（对应 README）
### 1) Bag of Word 与 N-gram 的性能差异
- BoW 最优（final 组）：`exp_02_bow_ce_lr0.2`，test_acc = **0.3321**
- N-gram 最优（final 组）：`exp_06_ngram_ce_lr0.2`，test_acc = **0.3203**

### 2) 不同损失函数、学习率的影响
- 见下方实验总览图（final 组网格实验）。
Figure 1：Task1 不同实验配置的测试集准确率对比图。横坐标表示实验名称，命名方式如 `exp_06_ngram_ce_lr0.2`，其中 `exp_06` 表示第 6 组实验，`ngram` 表示文本特征使用 N-gram，`ce` 表示损失函数为交叉熵，`lr0.2` 表示学习率为 0.2；纵坐标表示测试集准确率 `test_acc`，柱子越高说明该实验效果越好。
![task1_summary](/home/df_05/A_fnlp/pytorch/outputs/task1/final/experiment_summary.png)

### 3) 当前最优模型结果
- 最优 run：`best_ngram2_ce_lr01_30k`，test_acc = **0.3714**，val_acc = 0.3552
Figure 2：Task1 当前最优模型的训练过程曲线图。横坐标表示训练轮数 `epoch`；纵坐标分成两部分，其中左侧通常表示损失 `loss`，右侧或另一子图表示准确率 `accuracy`。通过这张图可以观察模型在训练集和验证集上是否持续收敛、是否出现平台期或过拟合。
![task1_best_curve](/home/df_05/A_fnlp/pytorch/outputs/task1/best/best_ngram2_ce_lr01_30k/training_curve.png)

## 我的思考
- 在当前数据上，线性模型上限有限，N-gram 比 BoW 略好但提升不大，说明仅靠表层共现特征表达能力不足。
- 训练后期验证集提升趋缓，继续堆迭代收益较小，更有效的是换模型结构（Task-2 的 RNN/Transformer）。
