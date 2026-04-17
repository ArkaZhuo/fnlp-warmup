from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from task3.train_addition import TrainConfig as AddTrainConfig
from task3.train_addition import run_training as run_addition


def parse_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Task-3 hyperparameter sweep for addition")
    parser.add_argument("--train-tsv", type=str, required=True)
    parser.add_argument("--test-tsv", type=str, required=True)
    parser.add_argument("--d-models", type=str, default="64,128")
    parser.add_argument("--nheads", type=str, default="2,4")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--save-dir", type=str, default="outputs/task3")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    d_models = parse_list(args.d_models)
    nheads = parse_list(args.nheads)

    base = AddTrainConfig(
        train_tsv=args.train_tsv,
        test_tsv=args.test_tsv,
        seed=42,
        max_src_len=16,
        max_tgt_len=16,
        d_model=128,
        nhead=4,
        enc_layers=2,
        dec_layers=2,
        ff_dim=256,
        dropout=0.1,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=1e-3,
        device=args.device,
        save_dir=args.save_dir,
        run_name=None,
        reverse_src=False,
        reverse_tgt=False,
    )

    rows = []
    run_id = 0
    for dm in d_models:
        for nh in nheads:
            run_id += 1
            run_name = f"exp_add_{run_id:02d}_d{dm}_h{nh}"
            cfg = replace(base, d_model=dm, nhead=nh, run_name=run_name)
            print(f"=== Running {run_name} ===")
            run_dir = run_addition(cfg)
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            rows.append({"run_name": run_name, "d_model": dm, "nhead": nh, "best_test_exact": metrics["best_test_exact"]})

    out_dir = Path(args.save_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parents[2] / out_dir
    out_path = out_dir / "addition_experiment_summary.json"
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary: {out_path}")


if __name__ == "__main__":
    main()
