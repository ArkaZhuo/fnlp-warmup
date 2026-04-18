# pytorch

这个目录是 `nlp-beginner` 相关任务的 PyTorch 实现，当前包含三个部分：

- `src/task1/`：基于手写线性分类器的文本分类
- `src/task2/`：基于 CNN / RNN / Transformer 的深度文本分类
- `src/task3/`：Transformer 基础结构与两个生成任务

这里只保留代码运行说明，不包含报告相关内容。

## 环境准备

先进入目录：

```bash
cd /home/df_05/A_fnlp/pytorch
```

建议使用当前环境：

```bash
conda activate nlp
```

默认数据文件已经放在：

```text
data/new_train.tsv
data/new_test.tsv
data/task3/add_train.tsv
data/task3/add_test.tsv
data/task3/lm_corpus.txt
```

## Task 1：传统文本分类

### 1. 训练

Bag-of-Words + 交叉熵：

```bash
/home/df_05/anaconda3/envs/nlp/bin/python src/task1/train.py \
  --feature-mode bow \
  --loss ce \
  --epochs 8 \
  --save-dir outputs/task1 \
  --run-name task1_bow_ce
```

N-gram + MSE：

```bash
/home/df_05/anaconda3/envs/nlp/bin/python src/task1/train.py \
  --feature-mode ngram \
  --ngram-n 2 \
  --loss mse \
  --epochs 8 \
  --save-dir outputs/task1 \
  --run-name task1_ngram_mse
```

### 2. 评估一个已有训练结果

```bash
/home/df_05/anaconda3/envs/nlp/bin/python src/task1/eval.py \
  --run-dir outputs/task1/task1_bow_ce
```

### 3. 批量实验

```bash
/home/df_05/anaconda3/envs/nlp/bin/python src/task1/experiments.py \
  --feature-modes bow,ngram \
  --losses ce,mse \
  --lrs 0.5,0.2 \
  --epochs 8 \
  --save-dir outputs/task1
```

## Task 2：深度学习文本分类

### 1. 训练 CNN

```bash
/home/df_05/anaconda3/envs/nlp/bin/python src/task2/train.py \
  --model-name cnn \
  --loss-name ce \
  --optimizer adam \
  --epochs 8 \
  --save-dir outputs/task2 \
  --run-name task2_cnn
```

### 2. 训练 RNN

```bash
/home/df_05/anaconda3/envs/nlp/bin/python src/task2/train.py \
  --model-name rnn \
  --loss-name ce \
  --optimizer adam \
  --epochs 8 \
  --save-dir outputs/task2 \
  --run-name task2_rnn
```

### 3. 训练 Transformer

```bash
/home/df_05/anaconda3/envs/nlp/bin/python src/task2/train.py \
  --model-name transformer \
  --loss-name ce \
  --optimizer adam \
  --epochs 8 \
  --save-dir outputs/task2 \
  --run-name task2_transformer
```

### 4. 使用 GloVe 初始化词向量

```bash
/home/df_05/anaconda3/envs/nlp/bin/python src/task2/train.py \
  --model-name cnn \
  --glove-path data/glove/glove.6B.50d.txt \
  --embed-dim 50 \
  --freeze-embedding \
  --epochs 8 \
  --save-dir outputs/task2 \
  --run-name task2_cnn_glove
```

### 5. 批量实验

```bash
/home/df_05/anaconda3/envs/nlp/bin/python src/task2/experiments.py \
  --models cnn,rnn,transformer \
  --losses ce,mse \
  --optimizers adam,sgd \
  --lrs 0.001,0.0005 \
  --epochs 4 \
  --save-dir outputs/task2
```

## Task 3：Transformer 基础结构

### 1. 生成加法数据

```bash
/home/df_05/anaconda3/envs/nlp/bin/python src/task3/data_gen.py add \
  --train-out data/task3/add_train.tsv \
  --test-out data/task3/add_test.tsv \
  --num-train 12000 \
  --num-test 2000 \
  --train-pairs "3+3,3+4,4+3" \
  --test-pairs "3+5,5+3,4+4"
```

### 2. 训练 Seq2Seq 加法模型

```bash
/home/df_05/anaconda3/envs/nlp/bin/python src/task3/train_addition.py \
  --train-tsv data/task3/add_train.tsv \
  --test-tsv data/task3/add_test.tsv \
  --epochs 12 \
  --reverse-src \
  --save-dir outputs/task3 \
  --run-name task3_addition
```

### 3. 做一次推理

```bash
/home/df_05/anaconda3/envs/nlp/bin/python src/task3/infer.py \
  --run-dir outputs/task3/task3_addition \
  --expr "123+456"
```

### 4. 批量超参数实验

```bash
/home/df_05/anaconda3/envs/nlp/bin/python src/task3/experiments.py \
  --train-tsv data/task3/add_train.tsv \
  --test-tsv data/task3/add_test.tsv \
  --d-models 64,128 \
  --nheads 2,4 \
  --epochs 8 \
  --save-dir outputs/task3
```

### 5. 训练 decoder-only 加法模型

```bash
/home/df_05/anaconda3/envs/nlp/bin/python src/task3/train_addition_decoder.py \
  --train-tsv data/task3/add_train.tsv \
  --test-tsv data/task3/add_test.tsv \
  --epochs 12 \
  --save-dir outputs/task3 \
  --run-name task3_addition_decoder
```

### 6. 生成语言模型语料

```bash
/home/df_05/anaconda3/envs/nlp/bin/python src/task3/data_gen.py lm \
  --train-tsv data/new_train.tsv \
  --out-path data/task3/lm_corpus.txt \
  --max-sentences 5000
```

### 7. 训练 decoder-only 语言模型

```bash
/home/df_05/anaconda3/envs/nlp/bin/python src/task3/train_lm.py \
  --corpus-path data/task3/lm_corpus.txt \
  --epochs 8 \
  --save-dir outputs/task3 \
  --run-name task3_lm
```

## 输出目录

训练结果默认会写到：

```text
outputs/task1/
outputs/task2/
outputs/task3/
```

每次训练通常会产出：

- `best_model.pt`
- `config.json`
- `metrics.json`
- `history.csv`
- 训练曲线图片
