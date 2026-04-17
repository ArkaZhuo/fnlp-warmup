# PyTorch基本练习

## Task1: 基于机器学习的文本分类

核心代码简要介绍：
文档对 Task1 的代码实现要求，核心是“用 PyTorch 只做张量和矩阵运算，不直接调用 `torch.nn`，自己完成文本向量化和 mini-batch 训练”。这一部分在实现上分成了四步。

第一步是数据读取。代码使用 `pandas` 读取 `new_train.tsv` 和 `new_test.tsv`，再通过 `train_test_split` 从训练集切出验证集，对应 `src/task1/data.py`。这样数据流就被明确拆成了训练、验证、测试三部分，满足 README 对数据划分的要求。

第二步是文本向量化。代码在 `src/task1/vectorizer.py` 中实现了 `Bag of Word` 和 `N-gram` 两种特征方式。具体做法是先分词，再统计 unigram 或 1 到 n 阶 gram 的出现频率，最后把每个句子表示成一个稀疏特征字典。后续训练前再把稀疏字典转成 batch 级别的稠密矩阵。这样就把 README 里“将句子转换为向量，即实现 Bag of Word 或者 N-gram”落到了代码里。

第三步是模型训练。代码在 `src/task1/model.py` 中实现了一个线性分类器，参数只有权重矩阵 `W` 和偏置 `b`。前向计算就是 `x @ W + b`。损失函数实现了两种：交叉熵和均方误差。交叉熵部分手写了 softmax、loss 和梯度；MSE 部分先把标签转成 one-hot，再计算误差和梯度。整个训练过程没有使用 `torch.nn.Module` 或现成优化器，而是直接用张量计算梯度并执行参数更新，这一点和 README 的要求是对齐的。

第四步是 mini-batch 训练与监控。代码在 `src/task1/train.py` 中把训练数据按 batch 切开，每个 batch 做一次前向、损失计算、梯度更新，再在每个 epoch 结束后跑一次验证集，记录 `train_loss/train_acc/val_loss/val_acc`，并把最优模型单独保存。最后再在测试集上评估。训练曲线和实验汇总图也都会保存下来，这对应 README 里“通过输出模型的 loss、在验证集上的正确率监测训练成果，并将结果绘制成图表”。

### 1. Bag of Word 与 N-gram 的性能差异
观察到的结果：
从总览图可以看出，在当前 `final` 这一组基础实验里，BoW 的最优结果略高于 N-gram。具体来说，BoW 最好的配置是 `exp_02_bow_ce_lr0.2`，测试准确率为 `0.3321`；N-gram 最好的配置是 `exp_06_ngram_ce_lr0.2`，测试准确率为 `0.3203`。这说明在当前基础词表规模和训练设置下，引入更高阶的局部搭配信息并没有直接带来收益，反而可能因为特征空间更大、更稀疏，使得线性模型更难稳定利用这些信息。

但如果看扩展实验，现象又有变化。当前 Task1 的全局最优 run 是 `best_ngram2_ce_lr01_30k`，测试准确率达到 `0.3714`。这说明 N-gram 不是天然比 BoW 差，而是它对词表大小、训练轮数和学习率更敏感。在更大的特征空间和更长训练下，二元语法信息开始发挥作用，因此最终最优模型仍然来自 N-gram。

从最优训练曲线还可以看到，训练损失在前期下降较快，但后期进入明显的平台区，验证集准确率提升有限。这说明线性模型本身的表达能力是主要瓶颈，而不仅仅是训练不充分。

如图：
Figure 1：Task1 基础实验总览图。横坐标表示不同实验配置名称，命名方式如 `exp_06_ngram_ce_lr0.2`，其中 `exp_06` 表示第 6 组实验，`ngram` 表示使用 N-gram 特征，`ce` 表示交叉熵损失，`lr0.2` 表示学习率 0.2；纵坐标表示测试集准确率 `test_acc`，柱子越高说明该配置分类效果越好。
![Task1实验总览](/home/df_05/A_fnlp/pytorch/outputs/task1/final/experiment_summary.png)
Figure 2：Task1 当前最优模型训练曲线图。横坐标表示训练轮数 `epoch`；纵坐标一部分表示训练集与验证集损失 `loss`，另一部分表示训练集与验证集准确率 `accuracy`。这张图主要用于观察模型的收敛速度以及后期是否进入平台期。
![Task1最优训练曲线](/home/df_05/A_fnlp/pytorch/outputs/task1/best/best_ngram2_ce_lr01_30k/training_curve.png)

### 2. 不同的损失函数、学习率对最终分类性能的影响
观察到的结果：
从图中各组柱子的相对高度可以看出，交叉熵整体稳定优于均方误差。这个现象是合理的，因为分类问题里交叉熵直接对应概率分布优化，优化目标与最终分类正确率更一致；而 MSE 更适合回归，在多分类场景下通常不如交叉熵敏感。

学习率方面，也能看到一个比较清楚的趋势：在 BoW 和 N-gram 两组实验中，`lr=0.2` 的结果普遍优于 `lr=0.5`。这说明 `0.5` 对当前这个线性模型来说偏大，更新幅度过猛，会让参数在较优区域附近震荡；而 `0.2` 更容易稳定收敛，因此得到更高的验证与测试表现。

换句话说，这张图反映出的不是单一结论，而是两层信息：一层是损失函数的选择会决定优化目标是否匹配分类任务；另一层是学习率太大时会损害训练稳定性。二者叠加后，`CE + 较小学习率` 成为 Task1 中更合理的组合。

如图：
Figure 3：Task1 不同损失函数与学习率组合的结果对比图。横坐标表示实验配置名称，命名规则与图1一致；纵坐标表示测试集准确率 `test_acc`。通过比较不同柱子的高度，可以直接看出 `BoW/N-gram`、`CE/MSE` 和不同学习率组合的差异。
![Task1损失函数和学习率影响](/home/df_05/A_fnlp/pytorch/outputs/task1/final/experiment_summary.png)


## Task2: 基于深度学习的文本分类

核心代码简要介绍：
文档对 Task2 的要求是：在 Task1 同样的数据基础上，把句子转成序列，使用 `embedding` 和深度学习模型做文本分类，并测试不同损失函数、学习率、卷积核、优化器、预训练词向量以及模型结构的影响。代码实现时基本按这个路线展开。

第一步仍然是数据处理。代码在 `src/task2/data.py` 中先读取 `new_train.tsv/new_test.tsv`，然后从训练集切验证集。与 Task1 不同的是，这一任务不再把句子转成稀疏向量，而是先做分词、建立词表，再把每个句子编码成定长 token 序列，不足的部分用 `<pad>` 补齐。这样就为后续的 `Embedding` 输入提供了标准化的整数序列。

第二步是词向量表示。README 明确提到“将句子转换为序列后直接调用 torch.nn.embedding 即可完成 embedding 操作”，代码也是这么实现的。`src/task2/models.py` 里的 CNN、BiLSTM 和 Transformer 分类器都以 `nn.Embedding` 作为第一层，把 token id 映射成稠密向量表示。这样模型学习的就不再是简单词频，而是带上下文建模能力的连续表示。

第三步是模型结构。CNN 模型通过一维卷积提取局部 n-gram 模式，再做 max pooling；BiLSTM 通过双向循环结构建模前后文依赖；Transformer 则使用自注意力机制做全局建模。这正对应 README 中“测试 CNN 改为 RNN、Transformer 等其它模型”的要求。三类模型都在 `src/task2/models.py` 中实现，并由 `src/task2/train.py` 统一训练和评估。

第四步是实验控制。`src/task2/train.py` 支持切换不同损失函数（CE/MSE）、优化器（Adam/SGD）、学习率、卷积核数量和大小，以及是否加载 GloVe 预训练向量。也就是说，README 里实验阶段要求的几组对比，代码都通过配置项被统一纳入了训练脚本，而不是临时手工修改模型。

### 1. CNN / RNN / Transformer 性能差异
观察到的结果：
从实验总览图可以看出，这三类模型在当前数据上的表现差距比较明显。RNN 的最佳结果是 `best_rnn_e8`，测试准确率达到 `0.4859`，是当前 Task2 的最好结果；Transformer 次之，为 `0.4554`；CNN 稍低，为 `0.4325`。这说明在该数据集上，句子的顺序信息和中长距离依赖关系是有价值的，单纯依赖卷积抽取局部模式不如循环模型有效。

从最优 RNN 的训练曲线还能看到一个典型现象：训练准确率持续快速上升，后期已经非常高，而验证准确率的提升在中后期开始变慢，验证损失甚至略有抬头。这说明模型后期开始出现一定过拟合，但由于保存的是验证集最优模型，因此最终测试表现仍然较强。

图里另外一个值得注意的现象是：Transformer 虽然不如 RNN 最终高，但已经明显超过 Task1 的线性模型，说明只要引入序列建模能力，即使结构不同，也能比传统向量化线性分类更有效。

如图：
Figure 4：Task2 基础实验总览图。横坐标表示实验配置名称，例如 `exp_03_rnn_ce_adam_lr0.001` 可以拆成“第 3 组实验 + RNN 模型 + 交叉熵损失 + Adam 优化器 + 学习率 0.001”；纵坐标表示测试集准确率 `test_acc`，用于比较不同模型结构和参数设置的效果。
![Task2实验总览](/home/df_05/A_fnlp/pytorch/outputs/task2/final/experiment_summary.png)
Figure 5：Task2 当前最优 RNN 模型训练曲线图。横坐标表示训练轮数 `epoch`；纵坐标表示损失 `loss` 和准确率 `accuracy` 的变化情况。通过训练曲线可以观察到训练准确率持续升高，而验证集提升后期放缓的现象。
![Task2最优训练曲线](/home/df_05/A_fnlp/pytorch/outputs/task2/best/best_rnn_e8/training_curve.png)

### 2. 不同损失函数、学习率及其它超参对性能的影响
观察到的结果：
先看损失函数和学习率。基于 CNN 的对照实验表明，`CE` 明显优于 `MSE`。例如在 Adam 优化器下，`CE + 1e-3` 的测试准确率是 `0.4261`，而 `MSE + 1e-3` 只有 `0.3950`。这和 Task1 的现象一致，说明对于分类任务，交叉熵依然是更自然的优化目标。学习率方面，`1e-3` 又普遍优于 `5e-4`，说明在当前深度模型规模下，`5e-4` 稍显保守，学习速度偏慢。

再看优化器和卷积核设置。Adam 与 SGD 的差距较大：在同样的 CNN 配置下，Adam 是 `0.4261`，SGD 只有 `0.3361`。这说明当前任务里自适应优化器更容易把模型推到较好的区域。卷积核个数从 64 增加到 128、或者把卷积核大小从 `[3,4,5]` 改成 `[2,3,4,5]`，都没有带来明显提升，反而略有下降，说明性能瓶颈不在于简单增加卷积容量。

最后看 GloVe。这里是最明显的一组对比：在 `embed_dim=50` 的同设置下，随机初始化时测试准确率为 `0.4056`，加载 GloVe 之后上升到 `0.4811`，而且确实成功加载了 7599 个预训练向量。从图里也能看出，GloVe 初始化的训练曲线下降更快、验证表现更高，这说明预训练词向量为模型提供了更好的初始语义空间，显著降低了从零开始学习表示的难度。

如图：
Figure 6：Task2 CNN 基线实验训练曲线图。横坐标表示训练轮数 `epoch`；纵坐标表示训练与验证的损失/准确率变化，用于观察 `CE + Adam + lr=1e-3` 这一基线配置的收敛过程。
![Task2-CE-Adam基线曲线](/home/df_05/A_fnlp/pytorch/outputs/task2/req/req_cnn_ce_adam_lr1e3_base/training_curve.png)
Figure 7：Task2 使用 GloVe 初始化后的训练曲线图。横坐标表示训练轮数 `epoch`；纵坐标表示训练与验证的损失/准确率变化，用于和图6进行对照，观察预训练词向量初始化后收敛是否更快、验证效果是否更高。
![Task2-GloVe曲线](/home/df_05/A_fnlp/pytorch/outputs/task2/req/req_cnn_ce_adam_lr1e3_glove50d/training_curve.png)


## Task3: 实现 Transformer 的基础结构

核心代码简要介绍：
文档对 Task3 的重点不是套现成分类器，而是“围绕 Transformer 自己搭建任务、生成数据、设计泛化实验，并区分训练和测试时的不同逻辑”。这一部分实现时分成了两个子任务：加法任务和语言模型任务。

在子任务1中，代码首先自己生成多位数加法数据，而不是使用现成数据集。`src/task3/data_gen.py` 支持指定训练集和测试集里允许出现的位数组合，例如 `3+3`、`3+4`、`4+4` 等，从而可以人为构造 easy、iid、base 等不同训练测试划分。这样就真正实现了 README 里要求的“自行设计不同实验验证模型泛化性”。

在模型结构上，代码在 `src/task3/models.py` 里实现了两种结构：一种是标准的 encoder-decoder Transformer，用于 seq2seq 加法任务；另一种是 decoder-only Transformer，用于做模型变种和语言模型实验。标准结构内部使用 `nn.Transformer`，包括 encoder、decoder、padding mask 和 causal mask；decoder-only 版本则通过 `TransformerEncoder` 加因果 mask 的方式实现自回归预测。

训练逻辑和测试逻辑是分开的。训练时，加法任务使用 teacher forcing，即把目标序列右移后整体送进 decoder，一次性预测所有位置；而测试时采用 greedy decoding，一步一步生成下一个 token，直到输出结束。这一点和 README 对 Transformer “训练和测试逻辑不同”的要求完全一致。

子任务2语言模型则基于情感分类训练文本构建字符级语料，用 decoder-only Transformer 做下一个字符预测。实验里又额外比较了不同的 `d_model` 和 `seq_len`，从而完成了 README 中“测试不同参数对模型训练效果的影响”和“自行构造一个语言模型”的要求。

### 1. 不同数据划分与模型变种的性能差异（子任务1）
观察到的结果：
从加法任务的几组结果可以很明显地看到，数据划分方式对模型泛化影响极大。当前最好的 easy 划分实验 `add_easy_rev_big` 的精确匹配率达到 `0.1483`，而 iid 划分下最好的 `add_iid_rev_finetune` 只有 `0.0158`，base 划分甚至接近于零。这说明模型在“训练分布内拟合”与“跨分布泛化”之间存在明显落差。

从最优 easy 曲线还能看出，训练损失在持续下降，但测试 exact match 上升较慢，且在中后期波动较大。这说明对加法这种组合泛化任务而言，loss 下降并不自动对应生成正确率大幅提升，模型可能学到了一部分局部规律，但还没有完全形成稳定的算法性泛化能力。

对比 seq2seq 和 decoder-only 也能看到结构差异。当前 easy 划分下，seq2seq 最好是 `0.1483`，而 decoder-only 版本 `add_easy_decoder_only` 只有 `0.0100`。这说明在当前任务设定下，把输入表达式和输出答案显式拆成 encoder-decoder 两部分，更利于模型学习对齐与条件生成；decoder-only 更难同时兼顾读入表达式和正确生成结果。

如图：
Figure 8：Task3 加法任务最优模型训练曲线图。横坐标表示训练轮数 `epoch`；纵坐标同时展示训练损失 `train_loss` 和测试集精确匹配率 `test_exact`。其中 `test_exact` 表示整条答案完全预测正确的比例，因此它比普通字符级准确率更严格。
![Task3加法最优曲线](/home/df_05/A_fnlp/pytorch/outputs/task3/best/add_easy_rev_big/curve.png)
Figure 9：Task3 加法任务 decoder-only 变种训练曲线图。横坐标表示训练轮数 `epoch`；纵坐标同样表示训练损失 `train_loss` 和测试精确匹配率 `test_exact`，用于和图8对比 seq2seq 与 decoder-only 在同类任务上的差异。
![Task3加法decoder-only曲线](/home/df_05/A_fnlp/pytorch/outputs/task3/add_easy_decoder_only/curve.png)

### 2. 不同参数对训练效果的影响（含子任务2语言模型）
观察到的结果：
语言模型实验里，可以看到参数调整确实带来了稳定收益。baseline 的 `val_ppl` 是 `10.4832`，而 `req_lm_d128_seq64` 降到了 `10.1624`，说明模型在验证集上的预测分布更集中、更准确。从训练曲线看，`val_loss` 和 `val_ppl` 在四个 epoch 内持续下降，没有出现明显反弹，说明当前还处在稳定改进阶段。

从最优 LM 曲线还能看出，下降速度前快后慢，说明模型已经学到了字符级局部统计规律，但继续优化的收益开始递减。生成样例里也能看到文本存在明显重复，例如 “the the the ...”，这说明虽然困惑度已经降低，但生成质量距离自然语言还差得很远。也就是说，这一组实验更说明模型具备了基本语言建模能力，而不是已经学到了高质量文本生成。

总体上，这组实验反映出两个现象：一是参数确实会改变训练效果；二是对 Transformer 来说，数值指标改善并不一定代表生成样例已经足够自然，因此在分析时必须同时看曲线、指标和样例文本。

如图：
Figure 10：Task3 语言模型最优实验训练曲线图。横坐标表示训练轮数 `epoch`；纵坐标表示训练损失 `train_loss` 与验证损失 `val_loss`，它们衡量模型对下一个字符预测的误差大小。曲线越往下说明模型拟合越充分。
![Task3-LM最优曲线](/home/df_05/A_fnlp/pytorch/outputs/task3/req/req_lm_d128_seq64/curve.png)
Figure 11：Task3 语言模型基线实验训练曲线图。横坐标表示训练轮数 `epoch`；纵坐标表示训练损失 `train_loss` 与验证损失 `val_loss`，用于和图10进行对照，比较不同参数设置下语言模型的收敛效果。
![Task3-LM基线曲线](/home/df_05/A_fnlp/pytorch/outputs/task3/lm_baseline/curve.png)


## 简短思考
- Task1 的实现更像是一个干净的基线实验，重点证明不用 `torch.nn` 也能用矩阵运算完整走通文本分类流程，但它的上限也很清楚。
- Task2 的实验说明，只要让模型真正看到序列结构和预训练语义信息，效果就会明显好于线性模型，因此这部分的核心不是“模型更复杂”，而是“表示能力更强”。
- Task3 最有价值的地方不在于当前分数有多高，而在于它把 Transformer 的训练方式、mask 机制、解码逻辑和泛化问题都真实暴露出来了。尤其是 easy 与 iid 的差距，说明模型离真正的算法泛化还有距离。
