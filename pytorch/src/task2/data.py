from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass
class TextDataset:
    texts: list[str]
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.texts)


@dataclass
class DatasetBundle:
    train: TextDataset
    val: TextDataset
    test: TextDataset


def load_tsv(path: str | Path) -> TextDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"])
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(np.int64).to_numpy()
    return TextDataset(texts=texts, labels=labels)


def split_train_val(dataset: TextDataset, val_ratio: float, seed: int) -> tuple[TextDataset, TextDataset]:
    x_tr, x_val, y_tr, y_val = train_test_split(
        dataset.texts,
        dataset.labels,
        test_size=val_ratio,
        random_state=seed,
        stratify=dataset.labels,
    )
    return (
        TextDataset(texts=x_tr, labels=np.asarray(y_tr, dtype=np.int64)),
        TextDataset(texts=x_val, labels=np.asarray(y_val, dtype=np.int64)),
    )


def load_bundle(train_path: str | Path, test_path: str | Path, val_ratio: float, seed: int) -> DatasetBundle:
    train_full = load_tsv(train_path)
    test = load_tsv(test_path)
    train, val = split_train_val(train_full, val_ratio=val_ratio, seed=seed)
    return DatasetBundle(train=train, val=val, test=test)


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    text = text.strip()
    if lowercase:
        text = text.lower()
    return text.split()


def build_vocab(
    texts: Iterable[str],
    min_freq: int = 2,
    max_size: int = 30000,
    lowercase: bool = True,
) -> dict[str, int]:
    counter: dict[str, int] = {}
    for text in texts:
        for token in tokenize(text, lowercase=lowercase):
            counter[token] = counter.get(token, 0) + 1
    kept = [(t, c) for t, c in counter.items() if c >= min_freq]
    kept.sort(key=lambda x: (-x[1], x[0]))
    kept = kept[: max_size - 2]
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for tok, _ in kept:
        vocab[tok] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int, lowercase: bool = True) -> list[int]:
    unk_id = vocab[UNK_TOKEN]
    tokens = tokenize(text, lowercase=lowercase)
    ids = [vocab.get(t, unk_id) for t in tokens]
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [vocab[PAD_TOKEN]] * (max_len - len(ids))


def encode_texts(texts: list[str], vocab: dict[str, int], max_len: int, lowercase: bool = True) -> np.ndarray:
    arr = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, text in enumerate(texts):
        arr[i] = np.asarray(encode_text(text, vocab, max_len=max_len, lowercase=lowercase), dtype=np.int64)
    return arr


def create_batches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    idx = np.arange(len(x))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(0, len(idx), batch_size):
        batch_idx = idx[i : i + batch_size]
        xb = torch.tensor(x[batch_idx], dtype=torch.long)
        yb = torch.tensor(y[batch_idx], dtype=torch.long)
        batches.append((xb, yb))
    return batches

