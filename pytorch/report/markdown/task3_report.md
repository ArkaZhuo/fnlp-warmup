# Task-3 Markdown Report（按 README 要求）

## 代码实现阶段
- 自行生成加法任务数据（`data/task3/add_*.tsv`），并构建语言模型语料（`lm_corpus.txt`）。
- 使用标准 Transformer 组件实现 seq2seq 与 decoder-only 两类结构。
- 训练与测试采用不同逻辑：训练 teacher forcing，测试自回归生成。

## 实验阶段（对应 README）
### 子任务1：多位数加法
- base 划分 best: `exp_add_02_d128_h2` -> exact=0.0005
- easy 划分 best: `add_easy_rev_big` -> exact=**0.1483**
- iid 划分 best: `add_iid_rev_finetune` -> exact=0.0158

### 子任务1：decoder-only 变种
- easy/seq2seq best: `add_easy_rev_big` -> exact=0.1483
- easy/decoder-only best: `add_easy_decoder_only` -> exact=0.0100

### 子任务2：语言模型
- `req_lm_d128_seq64`: val_ppl=10.1624 (d_model=128, seq_len=64)
- `lm_baseline`: val_ppl=10.4832 (d_model=128, seq_len=96)
- `req_lm_d64_seq96`: val_ppl=10.9120 (d_model=64, seq_len=96)
- 当前最好 LM: `req_lm_d128_seq64` -> **val_ppl=10.1624**

### 图表展示
Figure 1：Task3 加法任务最优模型的训练曲线图。横坐标表示训练轮数 `epoch`；纵坐标包含训练损失 `train_loss` 和测试精确匹配率 `test_exact`。其中 `test_exact` 表示模型生成的整个答案与标准答案完全一致的比例，因此比单个字符准确率更严格。
![task3_add_best_curve](/home/df_05/A_fnlp/pytorch/outputs/task3/best/add_easy_rev_big/curve.png)
Figure 2：Task3 语言模型最优实验的训练曲线图。横坐标表示训练轮数 `epoch`；纵坐标主要表示训练损失 `train_loss` 和验证损失 `val_loss`，它们反映模型对下一个字符预测的拟合程度。若验证损失持续下降，说明语言模型仍在稳定改进。
![task3_lm_best_curve](/home/df_05/A_fnlp/pytorch/outputs/task3/req/req_lm_d128_seq64/curve.png)

### 语言模型生成样例（截断）
> this and the the the the the the the the sthe the the the the the the the sthe the the the the the sthe the the the the the t

## 我的思考
- 加法任务上，数据划分方式对泛化影响极大；easy 显著高于 base/iid，说明模型在跨分布泛化上仍弱。
- 反转输入/输出（reverse_src/reverse_tgt）对加法任务有帮助，本质上减轻了长距离对齐难度。
- LM 的困惑度已经下降到较稳定区间，但生成文本仍有语义破碎，后续可尝试更大词表或更长训练。
