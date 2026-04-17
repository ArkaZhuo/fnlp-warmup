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
import torch.nn as nn

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from task2.data import PAD_TOKEN, build_vocab, create_batches, encode_texts, load_bundle
from task2.models import build_model


@dataclass
class TrainConfig:
    train_path: str
    test_path: str
    model_name: str
    val_ratio: float
    seed: int
    min_freq: int
    vocab_size: int
    max_len: int
    embed_dim: int
    batch_size: int
    epochs: int
    lr: float
    optimizer: str
    loss_name: str
    weight_decay: float
    num_kernels: int
    kernel_sizes: tuple[int, ...]
    hidden_dim: int
    nhead: int
    num_layers: int
    ff_dim: int
    dropout: float
    glove_path: str | None
    freeze_embedding: bool
    device: str
    save_dir: str
    run_name: str | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def maybe_load_glove(
    embedding: nn.Embedding,
    vocab: dict[str, int],
    glove_path: str | None,
    freeze: bool,
) -> int:
    if not glove_path:
        return 0
    path = Path(glove_path)
    if not path.exists():
        print(f"[Warn] GloVe file not found: {path}, skip initialization.")
        return 0
    loaded = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < embedding.embedding_dim + 1:
                continue
            token = parts[0]
            idx = vocab.get(token)
            if idx is None:
                continue
            vec = np.asarray(parts[1 : 1 + embedding.embedding_dim], dtype=np.float32)
            if vec.shape[0] != embedding.embedding_dim:
                continue
            embedding.weight.data[idx] = torch.tensor(vec, dtype=torch.float32)
            loaded += 1
    if freeze:
        embedding.weight.requires_grad = False
    return loaded


def compute_metrics(
    logits: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    loss_name: str,
    num_classes: int,
) -> tuple[torch.Tensor, int]:
    if loss_name == "ce":
        loss = loss_fn(logits, y)
    elif loss_name == "mse":
        target = torch.zeros((y.shape[0], num_classes), device=logits.device, dtype=torch.float32)
        target.scatter_(1, y.unsqueeze(1), 1.0)
        loss = loss_fn(logits, target)
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")
    pred = logits.argmax(dim=1)
    correct = int((pred == y).sum().item())
    return loss, correct


def evaluate(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    loss_fn: nn.Module,
    loss_name: str,
    num_classes: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for xb, yb in create_batches(x, y, batch_size=batch_size, shuffle=False, seed=0):
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss, correct = compute_metrics(
                logits,
                yb,
                loss_fn=loss_fn,
                loss_name=loss_name,
                num_classes=num_classes,
            )
            total_loss += float(loss.item()) * yb.numel()
            total_correct += correct
            total_count += int(yb.numel())
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def save_history(history: list[dict[str, float]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[dict[str, float]], out_png: Path) -> None:
    mpl_cache = out_png.parent / ".mplconfig"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    font_cache = out_png.parent / ".cache"
    font_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(font_cache))
    import matplotlib.pyplot as plt

    epochs = [int(h["epoch"]) for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run_training(cfg: TrainConfig) -> Path:
    set_seed(cfg.seed)
    project_root = Path(__file__).resolve().parents[2]
    save_root = Path(cfg.save_dir)
    if not save_root.is_absolute():
        save_root = project_root / save_root
    run_name = cfg.run_name or f"{cfg.model_name}_{cfg.loss_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = save_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    bundle = load_bundle(cfg.train_path, cfg.test_path, val_ratio=cfg.val_ratio, seed=cfg.seed)
    vocab = build_vocab(bundle.train.texts, min_freq=cfg.min_freq, max_size=cfg.vocab_size)
    pad_idx = vocab[PAD_TOKEN]

    x_train = encode_texts(bundle.train.texts, vocab=vocab, max_len=cfg.max_len)
    x_val = encode_texts(bundle.val.texts, vocab=vocab, max_len=cfg.max_len)
    x_test = encode_texts(bundle.test.texts, vocab=vocab, max_len=cfg.max_len)

    y_train = bundle.train.labels
    y_val = bundle.val.labels
    y_test = bundle.test.labels
    num_classes = int(np.max(y_train)) + 1

    model = build_model(
        model_name=cfg.model_name,
        vocab_size=len(vocab),
        num_classes=num_classes,
        embed_dim=cfg.embed_dim,
        pad_idx=pad_idx,
        max_len=cfg.max_len,
        num_kernels=cfg.num_kernels,
        kernel_sizes=cfg.kernel_sizes,
        hidden_dim=cfg.hidden_dim,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
    ).to(device)

    loaded_glove = 0
    if hasattr(model, "embedding"):
        loaded_glove = maybe_load_glove(
            embedding=model.embedding,  # type: ignore[attr-defined]
            vocab=vocab,
            glove_path=cfg.glove_path,
            freeze=cfg.freeze_embedding,
        )

    if cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    if cfg.loss_name == "ce":
        loss_fn = nn.CrossEntropyLoss()
    elif cfg.loss_name == "mse":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss: {cfg.loss_name}")

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        batches = create_batches(
            x_train,
            y_train,
            batch_size=cfg.batch_size,
            shuffle=True,
            seed=cfg.seed + epoch,
        )
        for xb, yb in batches:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss, correct = compute_metrics(
                logits,
                yb,
                loss_fn=loss_fn,
                loss_name=cfg.loss_name,
                num_classes=num_classes,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * yb.numel()
            total_correct += correct
            total_count += int(yb.numel())

        train_loss = total_loss / max(total_count, 1)
        train_acc = total_correct / max(total_count, 1)
        val_loss, val_acc = evaluate(
            model=model,
            x=x_val,
            y=y_val,
            batch_size=cfg.batch_size,
            loss_fn=loss_fn,
            loss_name=cfg.loss_name,
            num_classes=num_classes,
            device=device,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }
        )
        print(
            f"[Epoch {epoch:02d}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))
    test_loss, test_acc = evaluate(
        model=model,
        x=x_test,
        y=y_test,
        batch_size=cfg.batch_size,
        loss_fn=loss_fn,
        loss_name=cfg.loss_name,
        num_classes=num_classes,
        device=device,
    )
    print(f"[Test] loss={test_loss:.4f} acc={test_acc:.4f}")

    (run_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    save_history(history, run_dir / "history.csv")
    plot_history(history, run_dir / "training_curve.png")
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "vocab_size": len(vocab),
                "num_classes": num_classes,
                "num_train": int(len(y_train)),
                "num_val": int(len(y_val)),
                "num_test": int(len(y_test)),
                "device_used": str(device),
                "loaded_glove_vectors": int(loaded_glove),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return run_dir


def parse_kernel_sizes(raw: str) -> tuple[int, ...]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("kernel sizes cannot be empty")
    return tuple(vals)


def parse_args() -> TrainConfig:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Task-2 deep text classifier training")
    parser.add_argument("--train-path", type=str, default=str(project_root / "data/new_train.tsv"))
    parser.add_argument("--test-path", type=str, default=str(project_root / "data/new_test.tsv"))
    parser.add_argument("--model-name", type=str, choices=["cnn", "rnn", "transformer"], default="cnn")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="adam")
    parser.add_argument("--loss-name", type=str, choices=["ce", "mse"], default="ce")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-kernels", type=int, default=64)
    parser.add_argument("--kernel-sizes", type=str, default="3,4,5")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--glove-path", type=str, default=None)
    parser.add_argument("--freeze-embedding", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default="outputs/task2")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    return TrainConfig(
        train_path=args.train_path,
        test_path=args.test_path,
        model_name=args.model_name,
        val_ratio=args.val_ratio,
        seed=args.seed,
        min_freq=args.min_freq,
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        optimizer=args.optimizer,
        loss_name=args.loss_name,
        weight_decay=args.weight_decay,
        num_kernels=args.num_kernels,
        kernel_sizes=parse_kernel_sizes(args.kernel_sizes),
        hidden_dim=args.hidden_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        glove_path=args.glove_path,
        freeze_embedding=args.freeze_embedding,
        device=args.device,
        save_dir=args.save_dir,
        run_name=args.run_name,
    )


def main() -> None:
    cfg = parse_args()
    run_dir = run_training(cfg)
    print(f"Saved artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
