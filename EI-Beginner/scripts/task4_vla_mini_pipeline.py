#!/usr/bin/env python3
"""Task 4: Mini VLA pipeline with Open-X-like data adapter.

This is a lightweight, runnable stand-in for VLA training:
- Generate synthetic multimodal robot data (image + language + action).
- Export an Open-X-like JSONL dataset.
- Train a tiny VLA model (vision + language -> action delta).
- Evaluate prediction error and task success rate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Task4Result:
    task_name: str
    seed: int
    train_size: int
    test_size: int
    train_mse: float
    test_mse: float
    test_success_rate: float
    training_curve: list[dict[str, float]]
    openx_dataset_jsonl: str


COLORS = {
    "red": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "green": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "blue": np.array([0.0, 0.0, 1.0], dtype=np.float32),
}
LOCATIONS = {
    "top-left": np.array([5, 5], dtype=np.float32),
    "top-right": np.array([27, 5], dtype=np.float32),
    "bottom-left": np.array([5, 27], dtype=np.float32),
    "bottom-right": np.array([27, 27], dtype=np.float32),
    "center": np.array([16, 16], dtype=np.float32),
}


class TinyVLA(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.cnn_out = 32 * 8 * 8

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lang_proj = nn.Linear(emb_dim, 64)

        self.head = nn.Sequential(
            nn.Linear(self.cnn_out + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # dx, dy (normalized)
        )

    def forward(self, img: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        v = self.cnn(img)
        t = self.emb(tok).mean(dim=1)
        t = F.relu(self.lang_proj(t))
        x = torch.cat([v, t], dim=1)
        return self.head(x)


def draw_square(img: np.ndarray, xy: np.ndarray, color: np.ndarray, radius: int = 2) -> None:
    x, y = int(xy[0]), int(xy[1])
    x0, x1 = max(0, x - radius), min(31, x + radius)
    y0, y1 = max(0, y - radius), min(31, y + radius)
    img[y0 : y1 + 1, x0 : x1 + 1] = color


def generate_sample(rng: np.random.Generator) -> dict:
    img = np.full((32, 32, 3), 0.10, dtype=np.float32)

    color_names = list(COLORS.keys())
    rng.shuffle(color_names)
    selected_colors = color_names[:2]

    positions: dict[str, np.ndarray] = {}
    used = set()
    for c in selected_colors:
        while True:
            xy = np.array([rng.integers(4, 28), rng.integers(4, 28)], dtype=np.float32)
            key = (int(xy[0] // 3), int(xy[1] // 3))
            if key not in used:
                used.add(key)
                positions[c] = xy
                break

    for cname, pos in positions.items():
        draw_square(img, pos, COLORS[cname], radius=2)

    target_color = selected_colors[int(rng.integers(0, 2))]
    loc_name = list(LOCATIONS.keys())[int(rng.integers(0, len(LOCATIONS)))]
    target_xy = LOCATIONS[loc_name]

    # Add faint target marker for VLM-style grounding.
    draw_square(img, target_xy, np.array([0.9, 0.9, 0.9], dtype=np.float32), radius=1)

    src = positions[target_color]
    delta = (target_xy - src) / 16.0  # normalized action

    instruction = f"move {target_color} block to {loc_name}"
    return {
        "image": img,
        "instruction": instruction,
        "src_xy": src,
        "target_xy": target_xy,
        "action": delta.astype(np.float32),
    }


def build_vocab(texts: list[str]) -> dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for t in texts:
        for tok in t.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int = 8) -> np.ndarray:
    toks = [vocab.get(tok, 1) for tok in text.split()[:max_len]]
    if len(toks) < max_len:
        toks = toks + [0] * (max_len - len(toks))
    return np.asarray(toks, dtype=np.int64)


def to_tensors(samples: list[dict], vocab: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    imgs = np.stack([s["image"].transpose(2, 0, 1) for s in samples], axis=0).astype(np.float32)
    toks = np.stack([encode_text(s["instruction"], vocab) for s in samples], axis=0)
    actions = np.stack([s["action"] for s in samples], axis=0).astype(np.float32)

    x_img = torch.from_numpy(imgs)
    x_tok = torch.from_numpy(toks)
    y = torch.from_numpy(actions)
    return x_img, x_tok, y


def eval_success(pred: torch.Tensor, samples: list[dict]) -> float:
    pred_np = pred.detach().cpu().numpy()
    ok = 0
    for i, s in enumerate(samples):
        src = s["src_xy"]
        target = s["target_xy"]
        pred_target = src + pred_np[i] * 16.0
        err = float(np.linalg.norm(pred_target - target))
        if err <= 5.0:
            ok += 1
    return ok / len(samples) if samples else 0.0


def export_openx_like(samples: list[dict], out_jsonl: Path) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            row = {
                "episode_id": i,
                "step_id": 0,
                "language_instruction": s["instruction"],
                "observation": {
                    "image_shape": [32, 32, 3],
                    "src_xy": [float(s["src_xy"][0]), float(s["src_xy"][1])],
                    "target_xy": [float(s["target_xy"][0]), float(s["target_xy"][1])],
                },
                "action": {
                    "delta_xy_norm": [float(s["action"][0]), float(s["action"][1])],
                    "action_space": "cartesian_delta_xy",
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task4 mini VLA training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_size", type=int, default=2400)
    parser.add_argument("--test_size", type=int, default=600)
    parser.add_argument("--output", default="results/task4_vla_result.json")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    train_samples = [generate_sample(rng) for _ in range(args.train_size)]
    test_samples = [generate_sample(rng) for _ in range(args.test_size)]

    vocab = build_vocab([s["instruction"] for s in train_samples])
    x_img_tr, x_tok_tr, y_tr = to_tensors(train_samples, vocab)
    x_img_te, x_tok_te, y_te = to_tensors(test_samples, vocab)

    model = TinyVLA(vocab_size=len(vocab), emb_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    n = x_img_tr.shape[0]
    batch_size = 128

    training_curve: list[dict[str, float]] = []
    epochs = 70
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n)
        x_img_tr = x_img_tr[perm]
        x_tok_tr = x_tok_tr[perm]
        y_tr = y_tr[perm]

        for i in range(0, n, batch_size):
            xb_img = x_img_tr[i : i + batch_size]
            xb_tok = x_tok_tr[i : i + batch_size]
            yb = y_tr[i : i + batch_size]

            pred = model(xb_img, xb_tok)
            loss = F.mse_loss(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            with torch.no_grad():
                pred_train_epoch = model(x_img_tr, x_tok_tr)
                pred_test_epoch = model(x_img_te, x_tok_te)
                training_curve.append(
                    {
                        "epoch": float(epoch),
                        "train_mse": float(F.mse_loss(pred_train_epoch, y_tr).item()),
                        "test_mse": float(F.mse_loss(pred_test_epoch, y_te).item()),
                        "test_success_rate": float(eval_success(pred_test_epoch, test_samples)),
                    }
                )

    with torch.no_grad():
        pred_train = model(x_img_tr, x_tok_tr)
        pred_test = model(x_img_te, x_tok_te)
        train_mse = float(F.mse_loss(pred_train, y_tr).item())
        test_mse = float(F.mse_loss(pred_test, y_te).item())
        test_success = float(eval_success(pred_test, test_samples))

    openx_jsonl = Path("results/task4_openx_like_dataset.jsonl")
    export_openx_like(train_samples + test_samples, openx_jsonl)

    result = Task4Result(
        task_name="Task4-VLA-MiniPipeline-OpenXLike",
        seed=args.seed,
        train_size=args.train_size,
        test_size=args.test_size,
        train_mse=train_mse,
        test_mse=test_mse,
        test_success_rate=test_success,
        training_curve=training_curve,
        openx_dataset_jsonl=str(openx_jsonl),
    )

    payload = asdict(result)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
