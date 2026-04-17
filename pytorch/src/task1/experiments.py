from __future__ import annotations

import argparse
import csv
import os
from dataclasses import replace
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from task1.train import TrainConfig, run_training


def save_summary(rows: list[dict[str, str | float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "feature_mode",
                "loss",
                "lr",
                "best_val_acc",
                "test_acc",
                "vocab_size",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict[str, str | float]], out_path: Path) -> None:
    mpl_cache = out_path.parent / ".mplconfig"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    font_cache = out_path.parent / ".cache"
    font_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(font_cache))
    import matplotlib.pyplot as plt

    labels = [str(r["run_name"]) for r in rows]
    vals = [float(r["test_acc"]) for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4))
    ax.bar(labels, vals)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Task-1 Experiment Summary")
    ax.set_ylim(0.0, 1.0)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task-1 experiment sweep")
    parser.add_argument("--feature-modes", type=str, default="bow,ngram")
    parser.add_argument("--losses", type=str, default="ce,mse")
    parser.add_argument("--lrs", type=str, default="0.5,0.2")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ngram-n", type=int, default=2)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=12000)
    parser.add_argument("--save-dir", type=str, default="outputs/task1")
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--tfidf", action="store_true", default=False)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    train_path = args.train_path or str(project_root / "data/new_train.tsv")
    test_path = args.test_path or str(project_root / "data/new_test.tsv")

    base_cfg = TrainConfig(
        train_path=train_path,
        test_path=test_path,
        feature_mode="bow",
        ngram_n=args.ngram_n,
        min_freq=args.min_freq,
        max_features=args.max_features,
        val_ratio=0.1,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=0.5,
        loss="ce",
        weight_decay=0.0,
        normalize=args.normalize,
        tfidf=args.tfidf,
        device=args.device,
        save_dir=args.save_dir,
        run_name=None,
    )

    feature_modes = parse_str_list(args.feature_modes)
    losses = parse_str_list(args.losses)
    lrs = parse_float_list(args.lrs)

    summary_rows: list[dict[str, str | float]] = []
    run_id = 0
    for fm in feature_modes:
        for loss in losses:
            for lr in lrs:
                run_id += 1
                run_name = f"exp_{run_id:02d}_{fm}_{loss}_lr{lr}"
                cfg = replace(base_cfg, feature_mode=fm, loss=loss, lr=lr, run_name=run_name)
                print(f"=== Running {run_name} ===")
                run_dir = run_training(cfg)
                metrics_path = run_dir / "metrics.json"
                metrics = {}
                if metrics_path.exists():
                    import json

                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                summary_rows.append(
                    {
                        "run_name": run_name,
                        "feature_mode": fm,
                        "loss": loss,
                        "lr": lr,
                        "best_val_acc": float(metrics.get("best_val_acc", 0.0)),
                        "test_acc": float(metrics.get("test_acc", 0.0)),
                        "vocab_size": float(metrics.get("vocab_size", 0)),
                    }
                )

    save_root = Path(args.save_dir)
    if not save_root.is_absolute():
        save_root = project_root / save_root
    summary_csv = save_root / "experiment_summary.csv"
    summary_fig = save_root / "experiment_summary.png"
    save_summary(summary_rows, summary_csv)
    plot_summary(summary_rows, summary_fig)
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved summary plot: {summary_fig}")


if __name__ == "__main__":
    main()
