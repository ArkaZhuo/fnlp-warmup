from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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
    """Load TSV with no header: text<TAB>label."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"])
    if df.shape[1] != 2:
        raise ValueError(f"Expected 2 columns in {path}, got {df.shape[1]}")
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(np.int64).to_numpy()
    return TextDataset(texts=texts, labels=labels)


def split_train_val(
    dataset: TextDataset,
    val_ratio: float = 0.1,
    random_seed: int = 42,
    stratify: bool = True,
) -> tuple[TextDataset, TextDataset]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")
    stratify_labels: Sequence[int] | None = dataset.labels if stratify else None
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        dataset.texts,
        dataset.labels,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=stratify_labels,
    )
    return (
        TextDataset(texts=train_texts, labels=np.asarray(train_labels, dtype=np.int64)),
        TextDataset(texts=val_texts, labels=np.asarray(val_labels, dtype=np.int64)),
    )


def load_bundle(
    train_path: str | Path,
    test_path: str | Path,
    val_ratio: float = 0.1,
    random_seed: int = 42,
) -> DatasetBundle:
    train_full = load_tsv(train_path)
    test = load_tsv(test_path)
    train, val = split_train_val(
        dataset=train_full,
        val_ratio=val_ratio,
        random_seed=random_seed,
        stratify=True,
    )
    return DatasetBundle(train=train, val=val, test=test)

