# fnlp-warmup

这个仓库当前保留三部分核心代码：

- `EI-Beginner/`：机器人与 embodied intelligence 相关练习
- `agent-beginner/`：Agent / NexAU / NexDR 两个练习
- `pytorch/`：NLP-Beginner 的 PyTorch 实现

下面只保留代码运行说明，不包含报告相关内容。

## 环境建议

推荐直接使用你当前的 `nlp` 环境：

```bash
conda activate nlp
```

如果你更习惯显式写解释器，也可以直接使用：

```bash
/home/df_05/anaconda3/envs/nlp/bin/python
```

## 仓库结构

```text
EI-Beginner/           机器人相关任务
agent-beginner/        Agent 练习
pytorch/               文本分类与 Transformer 练习
```

## 运行入口

### 1. EI-Beginner

查看完整说明：

```bash
sed -n '1,240p' EI-Beginner/README.md
```

一个最常用的运行示例：

```bash
cd /home/df_05/A_fnlp/EI-Beginner
/home/df_05/anaconda3/envs/nlp/bin/python scripts/task1_pybullet_kinematic_pick_place.py --speed ultrafast
```

### 2. agent-beginner

查看完整说明：

```bash
sed -n '1,260p' agent-beginner/README.md
```

练习一终端 Demo：

```bash
cd /home/df_05/A_fnlp/agent-beginner
/home/df_05/anaconda3/envs/nlp/bin/python -m exercise1_nexau.run_terminal_demo
```

练习一网页 Demo：

```bash
cd /home/df_05/A_fnlp/agent-beginner
/home/df_05/anaconda3/envs/nlp/bin/python -m exercise1_nexau.demo_web --host 127.0.0.1 --port 8008
```

练习二运行：

```bash
cd /home/df_05/A_fnlp/agent-beginner
/home/df_05/anaconda3/envs/nlp/bin/python -m exercise2_nexdr.run_exercise2 \
  --query "How to improve RAG factuality in production?" \
  --inputs "exercise2_nexdr/samples/sample_notes.md,exercise2_nexdr/samples/sample_pdf.pdf,exercise2_nexdr/samples/sample_image.png" \
  --output_dir "exercise2_nexdr/workspaces/demo_v1"
```

### 3. pytorch

查看完整说明：

```bash
sed -n '1,260p' pytorch/README.md
```

Task 1 训练示例：

```bash
cd /home/df_05/A_fnlp/pytorch
/home/df_05/anaconda3/envs/nlp/bin/python src/task1/train.py --epochs 8 --feature-mode bow --loss ce
```

Task 2 训练示例：

```bash
cd /home/df_05/A_fnlp/pytorch
/home/df_05/anaconda3/envs/nlp/bin/python src/task2/train.py --model-name cnn --epochs 8
```

Task 3 训练示例：

```bash
cd /home/df_05/A_fnlp/pytorch
/home/df_05/anaconda3/envs/nlp/bin/python src/task3/train_addition.py --epochs 12 --reverse-src
```
