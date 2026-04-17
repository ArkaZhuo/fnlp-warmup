from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from task1.data import load_bundle
from task1.model import LinearClassifierScratch, sparse_batch_to_dense
from task1.vectorizer import NgramVectorizer


@dataclass
class TrainConfig:
    train_path: str
    test_path: str
    feature_mode: str
    ngram_n: int
    min_freq: int
    max_features: int
    val_ratio: float
    seed: int
    batch_size: int
    epochs: int
    lr: float
    loss: str
    weight_decay: float
    normalize: bool
    tfidf: bool
    device: str
    save_dir: str
    run_name: str | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batch_index_iter(
    n_items: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    indices = np.arange(n_items)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for i in range(0, n_items, batch_size):
        yield indices[i : i + batch_size]


def evaluate(
    model: LinearClassifierScratch,
    features: list[dict[int, float]],
    labels: np.ndarray,
    batch_size: int,
    loss_name: str,
    normalize: bool,
) -> tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    device = model.device

    for batch_idx in batch_index_iter(len(features), batch_size, shuffle=False, seed=0):
        batch_samples = [features[i] for i in batch_idx]
        x = sparse_batch_to_dense(
            batch_samples,
            input_dim=model.input_dim,
            device=device,
            l1_normalize=normalize,
        )
        y = torch.tensor(labels[batch_idx], dtype=torch.long, device=device)
        metrics = model.eval_batch(x, y, loss_name=loss_name)
        total_loss += metrics.loss * metrics.total
        total_correct += metrics.correct
        total_count += metrics.total

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


def save_history_csv(history: list[dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[dict[str, float]], out_path: Path) -> None:
    mpl_cache = out_path.parent / ".mplconfig"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    font_cache = out_path.parent / ".cache"
    font_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(font_cache))
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    train_acc = [float(row["train_acc"]) for row in history]
    val_acc = [float(row["val_acc"]) for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_training(cfg: TrainConfig) -> Path:
    set_seed(cfg.seed)

    project_root = Path(__file__).resolve().parents[2]
    save_root = Path(cfg.save_dir)
    if not save_root.is_absolute():
        save_root = project_root / save_root

    run_name = cfg.run_name
    if not run_name:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{cfg.feature_mode}_{cfg.loss}_{now}"
    run_dir = save_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)

    bundle = load_bundle(
        train_path=cfg.train_path,
        test_path=cfg.test_path,
        val_ratio=cfg.val_ratio,
        random_seed=cfg.seed,
    )

    vectorizer = NgramVectorizer(
        mode=cfg.feature_mode,
        ngram_n=cfg.ngram_n,
        min_freq=cfg.min_freq,
        max_features=cfg.max_features,
        lowercase=True,
    )
    vectorizer.fit(bundle.train.texts)
    weighting = "tfidf" if cfg.tfidf else "count"
    train_features = vectorizer.transform(bundle.train.texts, weighting=weighting)
    val_features = vectorizer.transform(bundle.val.texts, weighting=weighting)
    test_features = vectorizer.transform(bundle.test.texts, weighting=weighting)

    num_classes = int(bundle.train.labels.max()) + 1
    model = LinearClassifierScratch(
        input_dim=vectorizer.vocab_size,
        num_classes=num_classes,
        device=device,
        seed=cfg.seed,
    )

    best_val_acc = -1.0
    best_epoch = -1
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch_idx in batch_index_iter(
            len(train_features),
            batch_size=cfg.batch_size,
            shuffle=True,
            seed=cfg.seed + epoch,
        ):
            batch_samples = [train_features[i] for i in batch_idx]
            x = sparse_batch_to_dense(
                batch_samples,
                input_dim=model.input_dim,
                device=device,
                l1_normalize=cfg.normalize,
            )
            y = torch.tensor(bundle.train.labels[batch_idx], dtype=torch.long, device=device)
            metrics = model.train_batch(
                x=x,
                y=y,
                lr=cfg.lr,
                loss_name=cfg.loss,
                weight_decay=cfg.weight_decay,
            )
            total_loss += metrics.loss * metrics.total
            total_correct += metrics.correct
            total_count += metrics.total

        train_loss = total_loss / max(total_count, 1)
        train_acc = total_correct / max(total_count, 1)
        val_loss, val_acc = evaluate(
            model=model,
            features=val_features,
            labels=bundle.val.labels,
            batch_size=cfg.batch_size,
            loss_name=cfg.loss,
            normalize=cfg.normalize,
        )

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        }
        history.append(row)

        print(
            f"[Epoch {epoch:02d}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            model.save(str(run_dir / "best_model.pt"))

    model.load(str(run_dir / "best_model.pt"))
    test_loss, test_acc = evaluate(
        model=model,
        features=test_features,
        labels=bundle.test.labels,
        batch_size=cfg.batch_size,
        loss_name=cfg.loss,
        normalize=cfg.normalize,
    )
    print(f"[Test] loss={test_loss:.4f} acc={test_acc:.4f}")

    vectorizer.save(run_dir / "vectorizer.json")
    save_history_csv(history, run_dir / "history.csv")
    plot_history(history, run_dir / "training_curve.png")

    metrics = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "num_train": len(bundle.train),
        "num_val": len(bundle.val),
        "num_test": len(bundle.test),
        "vocab_size": vectorizer.vocab_size,
        "num_classes": num_classes,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return run_dir


def parse_args() -> TrainConfig:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Task-1 scratch linear text classifier")
    parser.add_argument("--train-path", type=str, default=str(project_root / "data/new_train.tsv"))
    parser.add_argument("--test-path", type=str, default=str(project_root / "data/new_test.tsv"))
    parser.add_argument("--feature-mode", type=str, choices=["bow", "ngram"], default="bow")
    parser.add_argument("--ngram-n", type=int, default=2)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=12000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--loss", type=str, choices=["ce", "mse"], default="ce")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--tfidf", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default="outputs/task1")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    return TrainConfig(
        train_path=args.train_path,
        test_path=args.test_path,
        feature_mode=args.feature_mode,
        ngram_n=args.ngram_n,
        min_freq=args.min_freq,
        max_features=args.max_features,
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        loss=args.loss,
        weight_decay=args.weight_decay,
        normalize=args.normalize,
        tfidf=args.tfidf,
        device=args.device,
        save_dir=args.save_dir,
        run_name=args.run_name,
    )


def main() -> None:
    cfg = parse_args()
    run_dir = run_training(cfg)
    print(f"Saved run artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
