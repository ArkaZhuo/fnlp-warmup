from __future__ import annotations

import nbformat as nbf
from pathlib import Path

ROOT = Path('/home/df_05/A_fnlp/pytorch')
REPORT_DIR = ROOT / 'report'
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def save_nb(path: Path, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    nb['metadata'] = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'},
    }
    path.write_text(nbf.writes(nb), encoding='utf-8')


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def task1_notebook() -> None:
    cells = [
        md(
            '# Task-1 Report: 基于机器学习的文本分类\n\n'
            '## 与 request.md 对齐清单\n'
            '- 使用 `new_train.tsv / new_test.tsv`，并从训练集切分验证集。\n'
            '- 特征对照：`BoW` vs `N-gram`。\n'
            '- 训练对照：不同损失函数（`CE/MSE`）与学习率。\n'
            '- 结果可视化：表格 + 图表。'
        ),
        md(
            '## 代码实现要点\n'
            '- 数据读取与划分：`src/task1/data.py`\n'
            '- BoW/N-gram 与 TF-IDF 向量化：`src/task1/vectorizer.py`\n'
            '- 纯张量线性分类器（不调用 `torch.nn`）：`src/task1/model.py`\n'
            '- mini-batch 训练、验证、测试与曲线输出：`src/task1/train.py`'
        ),
        code(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "ROOT = Path('/home/df_05/A_fnlp/pytorch')\n"
            "\n"
            "rows = []\n"
            "for p in (ROOT / 'outputs/task1').glob('**/metrics.json'):\n"
            "    cfg_p = p.parent / 'config.json'\n"
            "    if not cfg_p.exists():\n"
            "        continue\n"
            "    m = json.loads(p.read_text(encoding='utf-8'))\n"
            "    c = json.loads(cfg_p.read_text(encoding='utf-8'))\n"
            "    rel = p.parent.relative_to(ROOT / 'outputs/task1')\n"
            "    group = rel.parts[0] if len(rel.parts) > 1 else 'root'\n"
            "    rows.append({\n"
            "        'run': p.parent.name,\n"
            "        'group': group,\n"
            "        'feature_mode': c.get('feature_mode'),\n"
            "        'loss': c.get('loss'),\n"
            "        'lr': c.get('lr'),\n"
            "        'ngram_n': c.get('ngram_n'),\n"
            "        'tfidf': bool(c.get('tfidf', False)),\n"
            "        'normalize': bool(c.get('normalize', False)),\n"
            "        'vocab_size': m.get('vocab_size'),\n"
            "        'best_val_acc': m.get('best_val_acc'),\n"
            "        'test_acc': m.get('test_acc'),\n"
            "        'path': str(p.parent),\n"
            "    })\n"
            "df = pd.DataFrame(rows).sort_values('test_acc', ascending=False).reset_index(drop=True)\n"
            "display(df[['run','group','feature_mode','loss','lr','tfidf','normalize','best_val_acc','test_acc']].head(20))\n"
            "best_row = df.iloc[0]\n"
            "print('Best run:', best_row['run'], 'test_acc=', round(float(best_row['test_acc']), 4))"
        ),
        code(
            "# 1) BoW vs N-gram\n"
            "df_base = df[(df['loss'] == 'ce') & (df['group'].isin(['final','best']))]\n"
            "cmp = df_base.groupby('feature_mode', as_index=False)['test_acc'].max().sort_values('test_acc', ascending=False)\n"
            "display(cmp)\n"
            "plt.figure(figsize=(5,3))\n"
            "plt.bar(cmp['feature_mode'], cmp['test_acc'])\n"
            "plt.ylim(0.0, 1.0)\n"
            "plt.title('Task1: BoW vs N-gram (best test acc)')\n"
            "plt.ylabel('test_acc')\n"
            "plt.show()"
        ),
        code(
            "# 2) 不同损失函数 + 学习率\n"
            "df_grid = df[df['group'] == 'final'].copy()\n"
            "pivot = df_grid.pivot_table(index='loss', columns='lr', values='test_acc', aggfunc='max')\n"
            "display(pivot)\n"
            "ax = pivot.plot(kind='bar', figsize=(6,3))\n"
            "ax.set_ylim(0.0, 1.0)\n"
            "ax.set_title('Task1: Loss/LR impact on test_acc')\n"
            "ax.set_ylabel('test_acc')\n"
            "plt.tight_layout(); plt.show()"
        ),
        code(
            "# 3) 最优实验曲线\n"
            "from PIL import Image\n"
            "best_path = Path(best_row['path'])\n"
            "print('Best path:', best_path)\n"
            "print(json.dumps(json.loads((best_path/'metrics.json').read_text()), indent=2, ensure_ascii=False))\n"
            "img = Image.open(best_path/'training_curve.png')\n"
            "plt.figure(figsize=(10,4)); plt.imshow(img); plt.axis('off'); plt.title('Best Task1 Training Curve'); plt.show()"
        ),
    ]
    save_nb(REPORT_DIR / 'task1_report.ipynb', cells)


def task2_notebook() -> None:
    cells = [
        md(
            '# Task-2 Report: 基于深度学习的文本分类\n\n'
            '## 与 request.md 对齐清单\n'
            '- 数据读取与划分沿用 Task-1。\n'
            '- 模型对照：`CNN / RNN / Transformer`。\n'
            '- 训练对照：不同损失函数、学习率、优化器。\n'
            '- CNN 结构对照：卷积核数量、卷积核大小。\n'
            '- GloVe 预训练 embedding 初始化对照。\n'
            '- 结果可视化：表格 + 图表。'
        ),
        md(
            '## 代码实现要点\n'
            '- 数据预处理与批处理：`src/task2/data.py`\n'
            '- 模型实现（CNN/RNN/Transformer）：`src/task2/models.py`\n'
            '- 训练入口与实验参数（含 GloVe 初始化）：`src/task2/train.py`\n'
            '- 多组实验脚本：`src/task2/experiments.py`'
        ),
        code(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "ROOT = Path('/home/df_05/A_fnlp/pytorch')\n"
            "\n"
            "rows = []\n"
            "for p in (ROOT / 'outputs/task2').glob('**/metrics.json'):\n"
            "    cfg_p = p.parent / 'config.json'\n"
            "    if not cfg_p.exists():\n"
            "        continue\n"
            "    m = json.loads(p.read_text(encoding='utf-8'))\n"
            "    c = json.loads(cfg_p.read_text(encoding='utf-8'))\n"
            "    rel = p.parent.relative_to(ROOT / 'outputs/task2')\n"
            "    group = rel.parts[0] if len(rel.parts) > 1 else 'root'\n"
            "    rows.append({\n"
            "        'run': p.parent.name,\n"
            "        'group': group,\n"
            "        'model_name': c.get('model_name'),\n"
            "        'loss_name': c.get('loss_name'),\n"
            "        'optimizer': c.get('optimizer'),\n"
            "        'lr': c.get('lr'),\n"
            "        'num_kernels': c.get('num_kernels'),\n"
            "        'kernel_sizes': str(c.get('kernel_sizes')),\n"
            "        'embed_dim': c.get('embed_dim'),\n"
            "        'glove_path': c.get('glove_path'),\n"
            "        'loaded_glove_vectors': m.get('loaded_glove_vectors', 0),\n"
            "        'best_val_acc': m.get('best_val_acc'),\n"
            "        'test_acc': m.get('test_acc'),\n"
            "        'path': str(p.parent),\n"
            "    })\n"
            "df = pd.DataFrame(rows).sort_values('test_acc', ascending=False).reset_index(drop=True)\n"
            "display(df[['run','group','model_name','loss_name','optimizer','lr','num_kernels','kernel_sizes','embed_dim','loaded_glove_vectors','test_acc']].head(25))\n"
            "best_row = df.iloc[0]\n"
            "print('Best run:', best_row['run'], 'test_acc=', round(float(best_row['test_acc']), 4))"
        ),
        code(
            "# 1) CNN/RNN/Transformer 对照（优先用 best 组）\n"
            "df_model = df[df['group'].isin(['best','final']) & (df['loss_name'] == 'ce') & (df['optimizer'] == 'adam')]\n"
            "model_cmp = df_model.groupby('model_name', as_index=False)['test_acc'].max().sort_values('test_acc', ascending=False)\n"
            "display(model_cmp)\n"
            "plt.figure(figsize=(6,3))\n"
            "plt.bar(model_cmp['model_name'], model_cmp['test_acc'])\n"
            "plt.ylim(0.0, 1.0)\n"
            "plt.title('Task2: Model comparison (best test acc)')\n"
            "plt.ylabel('test_acc')\n"
            "plt.show()"
        ),
        code(
            "# 2) Loss + LR 对照（req 里的 CNN baseline）\n"
            "cond = (df['group'] == 'req') & (df['model_name'] == 'cnn') & (df['optimizer'] == 'adam') & (df['num_kernels'] == 64) & (df['kernel_sizes'] == '[3, 4, 5]') & (df['embed_dim'] == 128) & (df['glove_path'].isna())\n"
            "df_loss_lr = df[cond].copy()\n"
            "pivot = df_loss_lr.pivot_table(index='loss_name', columns='lr', values='test_acc', aggfunc='max')\n"
            "display(df_loss_lr[['run','loss_name','lr','test_acc']].sort_values(['loss_name','lr']))\n"
            "display(pivot)\n"
            "ax = pivot.plot(kind='bar', figsize=(6,3))\n"
            "ax.set_ylim(0.0, 1.0)\n"
            "ax.set_title('Task2: Loss/LR impact (CNN)')\n"
            "ax.set_ylabel('test_acc')\n"
            "plt.tight_layout(); plt.show()"
        ),
        code(
            "# 3) 优化器对照（同样配置下 Adam vs SGD）\n"
            "cond = (df['group'] == 'req') & (df['model_name'] == 'cnn') & (df['loss_name'] == 'ce') & (df['lr'] == 0.001) & (df['num_kernels'] == 64) & (df['kernel_sizes'] == '[3, 4, 5]') & (df['embed_dim'] == 128) & (df['glove_path'].isna())\n"
            "df_opt = df[cond][['run','optimizer','test_acc']].sort_values('optimizer')\n"
            "display(df_opt)\n"
            "plt.figure(figsize=(5,3))\n"
            "plt.bar(df_opt['optimizer'], df_opt['test_acc'])\n"
            "plt.ylim(0.0, 1.0)\n"
            "plt.title('Task2: Optimizer impact (CNN, CE, lr=1e-3)')\n"
            "plt.ylabel('test_acc')\n"
            "plt.show()"
        ),
        code(
            "# 4) 卷积核个数/大小对照\n"
            "cond = (df['group'] == 'req') & (df['model_name'] == 'cnn') & (df['loss_name'] == 'ce') & (df['optimizer'] == 'adam') & (df['lr'] == 0.001) & (df['embed_dim'] == 128) & (df['glove_path'].isna())\n"
            "df_kernel = df[cond][['run','num_kernels','kernel_sizes','test_acc']].drop_duplicates().sort_values('test_acc', ascending=False)\n"
            "display(df_kernel)\n"
            "labels = [f\"k={r.num_kernels},ks={r.kernel_sizes}\" for _, r in df_kernel.iterrows()]\n"
            "plt.figure(figsize=(8,3))\n"
            "plt.bar(labels, df_kernel['test_acc'])\n"
            "plt.ylim(0.0, 1.0)\n"
            "plt.xticks(rotation=20, ha='right')\n"
            "plt.title('Task2: Kernel setting impact (CNN)')\n"
            "plt.ylabel('test_acc')\n"
            "plt.tight_layout(); plt.show()"
        ),
        code(
            "# 5) GloVe 初始化对照（同 embed_dim=50）\n"
            "cond = (df['group'] == 'req') & (df['model_name'] == 'cnn') & (df['loss_name'] == 'ce') & (df['optimizer'] == 'adam') & (df['lr'] == 0.001) & (df['embed_dim'] == 50)\n"
            "df_glove = df[cond][['run','glove_path','loaded_glove_vectors','test_acc']].copy()\n"
            "df_glove['init'] = df_glove['glove_path'].apply(lambda x: 'glove' if isinstance(x, str) and len(x) > 0 else 'random')\n"
            "display(df_glove[['run','init','loaded_glove_vectors','test_acc']].sort_values('init'))\n"
            "plot_df = df_glove.groupby('init', as_index=False)['test_acc'].max()\n"
            "plt.figure(figsize=(5,3))\n"
            "plt.bar(plot_df['init'], plot_df['test_acc'])\n"
            "plt.ylim(0.0, 1.0)\n"
            "plt.title('Task2: GloVe init impact (embed_dim=50)')\n"
            "plt.ylabel('test_acc')\n"
            "plt.show()"
        ),
        code(
            "# 6) 当前最优结果与训练曲线\n"
            "from PIL import Image\n"
            "best_path = Path(best_row['path'])\n"
            "print('Best path:', best_path)\n"
            "print(json.dumps(json.loads((best_path/'metrics.json').read_text()), indent=2, ensure_ascii=False))\n"
            "img = Image.open(best_path/'training_curve.png')\n"
            "plt.figure(figsize=(10,4)); plt.imshow(img); plt.axis('off'); plt.title('Best Task2 Training Curve'); plt.show()"
        ),
    ]
    save_nb(REPORT_DIR / 'task2_report.ipynb', cells)


def task3_notebook() -> None:
    cells = [
        md(
            '# Task-3 Report: Transformer 基础结构\n\n'
            '## 与 request.md 对齐清单\n'
            '- 子任务1：多位数加法（自行生成数据）。\n'
            '- 子任务1：不同训练/测试划分验证泛化性（`easy / iid / 原始`）。\n'
            '- 子任务1：尝试 decoder-only 变种。\n'
            '- 子任务2：语言模型（自行准备语料），并做参数对照。\n'
            '- 结果可视化：表格 + 图表。'
        ),
        md(
            '## 代码实现要点\n'
            '- 数据构造：`src/task3/data_gen.py`\n'
            '- 编码器-解码器 Transformer（加法任务）：`src/task3/train_addition.py`\n'
            '- decoder-only 变种（加法）：`src/task3/train_addition_decoder.py`\n'
            '- decoder-only 语言模型：`src/task3/train_lm.py`'
        ),
        code(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "ROOT = Path('/home/df_05/A_fnlp/pytorch')\n"
            "\n"
            "# ---- Addition runs ----\n"
            "add_rows = []\n"
            "for p in (ROOT / 'outputs/task3').glob('**/metrics.json'):\n"
            "    m = json.loads(p.read_text(encoding='utf-8'))\n"
            "    if 'best_test_exact' not in m:\n"
            "        continue\n"
            "    cfg_p = p.parent / 'config.json'\n"
            "    c = json.loads(cfg_p.read_text(encoding='utf-8')) if cfg_p.exists() else {}\n"
            "    rel = p.parent.relative_to(ROOT / 'outputs/task3')\n"
            "    group = rel.parts[0] if len(rel.parts) > 1 else 'root'\n"
            "    run = p.parent.name\n"
            "    split = 'easy' if 'easy' in run or 'easy' in str(c.get('train_tsv', '')) else ('iid' if 'iid' in run or 'iid' in str(c.get('train_tsv', '')) else 'base')\n"
            "    model_variant = 'decoder_only' if 'decoder' in run else 'seq2seq'\n"
            "    add_rows.append({\n"
            "        'run': run,\n"
            "        'group': group,\n"
            "        'split': split,\n"
            "        'model_variant': model_variant,\n"
            "        'd_model': c.get('d_model'),\n"
            "        'nhead': c.get('nhead'),\n"
            "        'reverse_src': c.get('reverse_src', False),\n"
            "        'reverse_tgt': c.get('reverse_tgt', False),\n"
            "        'best_test_exact': m.get('best_test_exact'),\n"
            "        'path': str(p.parent),\n"
            "    })\n"
            "df_add = pd.DataFrame(add_rows).sort_values('best_test_exact', ascending=False).reset_index(drop=True)\n"
            "display(df_add[['run','group','split','model_variant','d_model','nhead','reverse_src','reverse_tgt','best_test_exact']].head(20))\n"
            "best_add = df_add.iloc[0]\n"
            "print('Best addition run:', best_add['run'], 'best_test_exact=', round(float(best_add['best_test_exact']), 4))"
        ),
        code(
            "# 1) 子任务1：不同划分（easy/iid/base）的泛化对照\n"
            "split_cmp = df_add.groupby('split', as_index=False)['best_test_exact'].max().sort_values('best_test_exact', ascending=False)\n"
            "display(split_cmp)\n"
            "plt.figure(figsize=(5,3))\n"
            "plt.bar(split_cmp['split'], split_cmp['best_test_exact'])\n"
            "plt.ylim(0.0, 1.0)\n"
            "plt.title('Task3-Add: split generalization (best exact)')\n"
            "plt.ylabel('exact_match')\n"
            "plt.show()"
        ),
        code(
            "# 2) 子任务1：decoder-only 变种对照（easy 划分）\n"
            "easy = df_add[df_add['split'] == 'easy']\n"
            "variant_cmp = easy.groupby('model_variant', as_index=False)['best_test_exact'].max().sort_values('best_test_exact', ascending=False)\n"
            "display(variant_cmp)\n"
            "plt.figure(figsize=(5,3))\n"
            "plt.bar(variant_cmp['model_variant'], variant_cmp['best_test_exact'])\n"
            "plt.ylim(0.0, 1.0)\n"
            "plt.title('Task3-Add: seq2seq vs decoder-only (easy)')\n"
            "plt.ylabel('exact_match')\n"
            "plt.show()"
        ),
        code(
            "# 3) 子任务1：参数/技巧影响（reverse 与更大模型）\n"
            "cols = ['run','split','d_model','nhead','reverse_src','reverse_tgt','best_test_exact']\n"
            "display(df_add[cols].sort_values('best_test_exact', ascending=False).head(12))"
        ),
        code(
            "# ---- LM runs ----\n"
            "lm_rows = []\n"
            "for p in (ROOT / 'outputs/task3').glob('**/metrics.json'):\n"
            "    m = json.loads(p.read_text(encoding='utf-8'))\n"
            "    if 'best_val_ppl' not in m:\n"
            "        continue\n"
            "    cfg_p = p.parent / 'config.json'\n"
            "    c = json.loads(cfg_p.read_text(encoding='utf-8')) if cfg_p.exists() else {}\n"
            "    rel = p.parent.relative_to(ROOT / 'outputs/task3')\n"
            "    group = rel.parts[0] if len(rel.parts) > 1 else 'root'\n"
            "    lm_rows.append({\n"
            "        'run': p.parent.name,\n"
            "        'group': group,\n"
            "        'd_model': c.get('d_model'),\n"
            "        'nhead': c.get('nhead'),\n"
            "        'seq_len': c.get('seq_len'),\n"
            "        'best_val_ppl': m.get('best_val_ppl'),\n"
            "        'best_val_loss': m.get('best_val_loss'),\n"
            "        'path': str(p.parent),\n"
            "    })\n"
            "df_lm = pd.DataFrame(lm_rows).sort_values('best_val_ppl').reset_index(drop=True)\n"
            "display(df_lm)\n"
            "best_lm = df_lm.iloc[0]\n"
            "print('Best LM run:', best_lm['run'], 'best_val_ppl=', round(float(best_lm['best_val_ppl']), 4))\n"
            "plt.figure(figsize=(6,3))\n"
            "plt.bar(df_lm['run'], df_lm['best_val_ppl'])\n"
            "plt.xticks(rotation=20, ha='right')\n"
            "plt.title('Task3-LM: parameter impact (val perplexity)')\n"
            "plt.ylabel('val_ppl (lower is better)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        code(
            "# 4) 样例生成文本（最佳 LM）\n"
            "best_lm_path = Path(best_lm['path'])\n"
            "sample_p = best_lm_path / 'sample.txt'\n"
            "if sample_p.exists():\n"
            "    txt = sample_p.read_text(encoding='utf-8')\n"
            "    print(txt[:1200])\n"
            "else:\n"
            "    print('No sample.txt found at', sample_p)"
        ),
        code(
            "# 5) 最佳加法模型曲线\n"
            "from PIL import Image\n"
            "best_add_path = Path(best_add['path'])\n"
            "print('Best addition path:', best_add_path)\n"
            "print(json.dumps(json.loads((best_add_path/'metrics.json').read_text()), indent=2, ensure_ascii=False))\n"
            "curve = best_add_path / 'curve.png'\n"
            "if curve.exists():\n"
            "    img = Image.open(curve)\n"
            "    plt.figure(figsize=(8,4)); plt.imshow(img); plt.axis('off'); plt.title('Best Task3 Addition Curve'); plt.show()\n"
            "else:\n"
            "    print('No curve found:', curve)"
        ),
    ]
    save_nb(REPORT_DIR / 'task3_report.ipynb', cells)


def main() -> None:
    task1_notebook()
    task2_notebook()
    task3_notebook()

    old_task4 = REPORT_DIR / 'task4_report.ipynb'
    if old_task4.exists():
        old_task4.unlink()

    print('Generated notebooks:')
    for p in sorted(REPORT_DIR.glob('task*_report.ipynb')):
        print(p)


if __name__ == '__main__':
    main()
