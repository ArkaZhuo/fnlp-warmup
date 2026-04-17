from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import replace
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from task2.train import TrainConfig, run_training


def parse_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def save_summary(rows: list[dict[str, str | float]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "model_name",
                "loss_name",
                "optimizer",
                "lr",
                "best_val_acc",
                "test_acc",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict[str, str | float]], out_png: Path) -> None:
    mpl_cache = out_png.parent / ".mplconfig"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    font_cache = out_png.parent / ".cache"
    font_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(font_cache))
    import matplotlib.pyplot as plt

    labels = [str(r["run_name"]) for r in rows]
    vals = [float(r["test_acc"]) for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4))
    ax.bar(labels, vals)
    ax.set_title("Task-2 Experiment Summary")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0.0, 1.0)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task-2 experiment sweep")
    parser.add_argument("--models", type=str, default="cnn,rnn,transformer")
    parser.add_argument("--losses", type=str, default="ce,mse")
    parser.add_argument("--optimizers", type=str, default="adam,sgd")
    parser.add_argument("--lrs", type=str, default="0.001,0.0005")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="outputs/task2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--max-len", type=int, default=80)
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    train_path = args.train_path or str(project_root / "data/new_train.tsv")
    test_path = args.test_path or str(project_root / "data/new_test.tsv")
    base = TrainConfig(
        train_path=train_path,
        test_path=test_path,
        model_name="cnn",
        val_ratio=0.1,
        seed=args.seed,
        min_freq=args.min_freq,
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=1e-3,
        optimizer="adam",
        loss_name="ce",
        weight_decay=0.0,
        num_kernels=64,
        kernel_sizes=(3, 4, 5),
        hidden_dim=128,
        nhead=4,
        num_layers=2,
        ff_dim=256,
        dropout=args.dropout,
        glove_path=None,
        freeze_embedding=False,
        device=args.device,
        save_dir=args.save_dir,
        run_name=None,
    )

    models = parse_list(args.models)
    losses = parse_list(args.losses)
    optimizers = parse_list(args.optimizers)
    lrs = parse_float_list(args.lrs)

    rows: list[dict[str, str | float]] = []
    run_id = 0
    for m in models:
        for loss in losses:
            for opt in optimizers:
                for lr in lrs:
                    run_id += 1
                    run_name = f"exp_{run_id:02d}_{m}_{loss}_{opt}_lr{lr}"
                    cfg = replace(
                        base,
                        model_name=m,
                        loss_name=loss,
                        optimizer=opt,
                        lr=lr,
                        run_name=run_name,
                    )
                    print(f"=== Running {run_name} ===")
                    run_dir = run_training(cfg)
                    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                    rows.append(
                        {
                            "run_name": run_name,
                            "model_name": m,
                            "loss_name": loss,
                            "optimizer": opt,
                            "lr": lr,
                            "best_val_acc": float(metrics["best_val_acc"]),
                            "test_acc": float(metrics["test_acc"]),
                        }
                    )

    save_root = Path(args.save_dir)
    if not save_root.is_absolute():
        save_root = project_root / save_root
    save_summary(rows, save_root / "experiment_summary.csv")
    plot_summary(rows, save_root / "experiment_summary.png")
    print(f"Saved summary to: {save_root}")


if __name__ == "__main__":
    main()

