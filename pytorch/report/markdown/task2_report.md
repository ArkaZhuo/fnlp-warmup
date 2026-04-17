# Task-2 Markdown Report（按 README 要求）

## 代码实现阶段
- 使用与 Task-1 相同数据划分流程。
- 句子转索引序列并通过 `nn.Embedding` 得到词向量。
- 实现并训练 CNN / BiLSTM / Transformer 三种分类模型。
- 增加 GloVe 预训练 embedding 初始化实验。

## 实验阶段（对应 README）
### 1) 不同损失函数、学习率
- CE, lr=1e-3: `req_cnn_ce_adam_lr1e3_base` -> **0.4261**
- CE, lr=5e-4: `req_cnn_ce_adam_lr5e4_base` -> **0.4170**
- MSE, lr=1e-3: `req_cnn_mse_adam_lr1e3_base` -> **0.3950**
- MSE, lr=5e-4: `req_cnn_mse_adam_lr5e4_base` -> **0.3747**

### 2) 卷积核个数、大小、优化器
- Optimizer 对照（同配置 CE,1e-3）：Adam=0.4261 vs SGD=0.3361
- 卷积核个数：64 -> 0.4261，128 -> 0.4198
- 卷积核大小：[3,4,5] -> 0.4261，[2,3,4,5] -> 0.4189

### 3) GloVe 初始化影响
- 随机初始化（50d）：0.4056
- GloVe 初始化（50d，loaded=7599）：**0.4811**

### 4) CNN 改为 RNN / Transformer
- CNN best: `best_cnn_e8` -> **0.4325**
- RNN best: `best_rnn_e8` -> **0.4859**
- TRANSFORMER best: `best_tf_e8` -> **0.4554**

### 5) 图表展示
Figure 1：Task2 不同模型与参数配置的测试集准确率对比图。横坐标表示实验名称，例如 `exp_03_rnn_ce_adam_lr0.001` 可拆成“第 3 组实验 + RNN 模型 + 交叉熵损失 + Adam 优化器 + 学习率 0.001”；纵坐标表示测试集准确率 `test_acc`，柱子越高说明该模型配置表现越好。
![task2_summary](/home/df_05/A_fnlp/pytorch/outputs/task2/final/experiment_summary.png)
Figure 2：Task2 当前最优 RNN 模型的训练曲线图。横坐标表示训练轮数 `epoch`；纵坐标一部分表示训练集和验证集损失 `loss`，另一部分表示训练集和验证集准确率 `accuracy`。这张图主要用来判断模型是否收敛，以及后期是否出现训练持续上升但验证提升变慢的过拟合迹象。
![task2_best_curve](/home/df_05/A_fnlp/pytorch/outputs/task2/best/best_rnn_e8/training_curve.png)

## 我的思考
- 同一数据上，深度模型显著优于 Task-1 线性模型，说明序列建模能力更关键。
- GloVe 在 50d 设置下带来明显提升，说明预训练语义先验对小样本监督任务帮助很大。
- SGD 在当前超参下明显弱于 Adam；后续若继续用 SGD，需单独调学习率与动量策略。
