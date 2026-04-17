from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))


@dataclass
class AdditionSample:
    src: str
    tgt: str


def random_int_with_digits(d: int) -> int:
    if d <= 1:
        return random.randint(0, 9)
    lo = 10 ** (d - 1)
    hi = 10**d - 1
    return random.randint(lo, hi)


def make_addition_dataset(
    num_samples: int,
    digit_pairs: list[tuple[int, int]],
    seed: int = 42,
) -> list[AdditionSample]:
    random.seed(seed)
    samples: list[AdditionSample] = []
    for _ in range(num_samples):
        d1, d2 = random.choice(digit_pairs)
        a = random_int_with_digits(d1)
        b = random_int_with_digits(d2)
        src = f"{a}+{b}"
        tgt = str(a + b)
        samples.append(AdditionSample(src=src, tgt=tgt))
    return samples


def save_addition_tsv(samples: list[AdditionSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{"src": s.src, "tgt": s.tgt} for s in samples])
    df.to_csv(path, sep="\t", index=False)


def load_sentiment_texts(train_tsv_path: Path) -> list[str]:
    df = pd.read_csv(train_tsv_path, sep="\t", header=None, names=["text", "label"])
    texts = df["text"].astype(str).tolist()
    return texts


def build_lm_corpus(
    train_tsv_path: Path,
    out_path: Path,
    max_sentences: int = 5000,
    seed: int = 42,
) -> None:
    texts = load_sentiment_texts(train_tsv_path)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(texts))
    rng.shuffle(idx)
    picked = [texts[i] for i in idx[: min(max_sentences, len(texts))]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(picked), encoding="utf-8")


def parse_digit_pairs(raw: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "+" not in chunk:
            raise ValueError(f"invalid pair: {chunk}")
        a, b = chunk.split("+", 1)
        pairs.append((int(a), int(b)))
    if not pairs:
        raise ValueError("digit pairs cannot be empty")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Task-3 data generator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="generate addition train/test TSV")
    p_add.add_argument("--train-out", type=str, required=True)
    p_add.add_argument("--test-out", type=str, required=True)
    p_add.add_argument("--num-train", type=int, default=12000)
    p_add.add_argument("--num-test", type=int, default=2000)
    p_add.add_argument("--train-pairs", type=str, default="3+3,3+4,4+3")
    p_add.add_argument("--test-pairs", type=str, default="3+5,5+3,4+4")
    p_add.add_argument("--seed", type=int, default=42)

    p_lm = sub.add_parser("lm", help="build LM corpus from sentiment train text")
    p_lm.add_argument("--train-tsv", type=str, required=True)
    p_lm.add_argument("--out-path", type=str, required=True)
    p_lm.add_argument("--max-sentences", type=int, default=5000)
    p_lm.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.cmd == "add":
        train_pairs = parse_digit_pairs(args.train_pairs)
        test_pairs = parse_digit_pairs(args.test_pairs)
        train_samples = make_addition_dataset(args.num_train, train_pairs, seed=args.seed)
        test_samples = make_addition_dataset(args.num_test, test_pairs, seed=args.seed + 1)
        save_addition_tsv(train_samples, Path(args.train_out))
        save_addition_tsv(test_samples, Path(args.test_out))
        print(f"Saved addition train/test to {args.train_out} and {args.test_out}")
    else:
        build_lm_corpus(
            train_tsv_path=Path(args.train_tsv),
            out_path=Path(args.out_path),
            max_sentences=args.max_sentences,
            seed=args.seed,
        )
        print(f"Saved LM corpus to {args.out_path}")


if __name__ == "__main__":
    main()

