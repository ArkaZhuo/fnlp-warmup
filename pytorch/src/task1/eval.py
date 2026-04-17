from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from task1.data import load_tsv
from task1.model import LinearClassifierScratch, sparse_batch_to_dense
from task1.train import batch_index_iter
from task1.vectorizer import NgramVectorizer


def evaluate_run(
    run_dir: Path,
    test_path: Path,
    batch_size: int = 128,
    normalize: bool | None = None,
    device: str = "cpu",
) -> dict[str, float]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    vectorizer = NgramVectorizer.load(run_dir / "vectorizer.json")

    dataset = load_tsv(test_path)
    weighting = "tfidf" if bool(config.get("tfidf", False)) else "count"
    features = vectorizer.transform(dataset.texts, weighting=weighting)

    normalize_flag = bool(config.get("normalize", False)) if normalize is None else normalize
    state = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    input_dim, num_classes = state["W"].shape
    model = LinearClassifierScratch(
        input_dim=int(input_dim),
        num_classes=num_classes,
        device=torch.device(device),
        seed=int(config.get("seed", 42)),
    )
    model.W = state["W"].to(model.device)
    model.b = state["b"].to(model.device)

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    loss_name = str(config.get("loss", "ce"))

    for batch_idx in batch_index_iter(len(features), batch_size, shuffle=False, seed=0):
        batch_samples = [features[i] for i in batch_idx]
        x = sparse_batch_to_dense(
            batch_samples,
            input_dim=model.input_dim,
            device=model.device,
            l1_normalize=normalize_flag,
        )
        y = torch.tensor(dataset.labels[batch_idx], dtype=torch.long, device=model.device)
        metrics = model.eval_batch(x, y, loss_name=loss_name)
        total_loss += metrics.loss * metrics.total
        total_correct += metrics.correct
        total_count += metrics.total

    result = {
        "loss": total_loss / max(total_count, 1),
        "acc": total_correct / max(total_count, 1),
        "count": float(total_count),
    }
    return result


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Evaluate a saved Task-1 run")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to outputs/task1/<run_name>")
    parser.add_argument("--test-path", type=str, default=str(project_root / "data/new_test.tsv"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--normalize",
        action="store_const",
        const=True,
        default=None,
        help="Force L1 normalization for input counts. Default uses training config.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    result = evaluate_run(
        run_dir=run_dir,
        test_path=Path(args.test_path).resolve(),
        batch_size=args.batch_size,
        normalize=args.normalize,
        device=args.device,
    )
    print(
        f"Run={run_dir.name} Test loss={result['loss']:.4f} "
        f"acc={result['acc']:.4f} n={int(result['count'])}"
    )


if __name__ == "__main__":
    main()
