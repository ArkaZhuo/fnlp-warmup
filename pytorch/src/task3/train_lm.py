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

from task3.models import DecoderOnlyTransformer
from task3.tokenizer import CharTokenizer


@dataclass
class TrainConfig:
    corpus_path: str
    seq_len: int
    stride: int
    seed: int
    d_model: int
    nhead: int
    num_layers: int
    ff_dim: int
    dropout: float
    batch_size: int
    epochs: int
    lr: float
    val_ratio: float
    device: str
    save_dir: str
    run_name: str | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_sequences(token_ids: list[int], seq_len: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    x_list: list[list[int]] = []
    y_list: list[list[int]] = []
    for i in range(0, len(token_ids) - seq_len - 1, stride):
        x = token_ids[i : i + seq_len]
        y = token_ids[i + 1 : i + seq_len + 1]
        x_list.append(x)
        y_list.append(y)
    x_arr = np.asarray(x_list, dtype=np.int64)
    y_arr = np.asarray(y_list, dtype=np.int64)
    return x_arr, y_arr


def split_train_val(x: np.ndarray, y: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(x))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(len(idx) * val_ratio)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]


def make_batches(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    idx = np.arange(len(x))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(0, len(idx), batch_size):
        b = idx[i : i + batch_size]
        out.append((torch.tensor(x[b], dtype=torch.long), torch.tensor(y[b], dtype=torch.long)))
    return out


@torch.no_grad()
def evaluate(
    model: DecoderOnlyTransformer,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tok = 0
    total_correct = 0
    for xb, yb in make_batches(x, y, batch_size=batch_size, shuffle=False, seed=0):
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == yb).sum().item())
        total_tok += int(yb.numel())
        total_loss += float(loss.item()) * int(yb.numel())
    return total_loss / max(total_tok, 1), total_correct / max(total_tok, 1)


@torch.no_grad()
def sample_text(
    model: DecoderOnlyTransformer,
    tokenizer: CharTokenizer,
    prefix: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    model.eval()
    ids = tokenizer.encode(prefix, add_bos=True, add_eos=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(x)
        nxt = int(logits[0, -1].argmax().item())
        x = torch.cat([x, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
        if nxt == tokenizer.eos_id:
            break
    return tokenizer.decode(x[0].tolist(), skip_special=True)


def run_training(cfg: TrainConfig) -> Path:
    set_seed(cfg.seed)
    project_root = Path(__file__).resolve().parents[2]
    save_root = Path(cfg.save_dir)
    if not save_root.is_absolute():
        save_root = project_root / save_root
    run_name = cfg.run_name or f"lm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = save_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    corpus_text = Path(cfg.corpus_path).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.build([corpus_text])
    ids = tokenizer.encode(corpus_text, add_bos=True, add_eos=True)
    x, y = build_sequences(ids, seq_len=cfg.seq_len, stride=cfg.stride)
    x_tr, y_tr, x_val, y_val = split_train_val(x, y, val_ratio=cfg.val_ratio, seed=cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        max_len=max(cfg.seq_len + 64, 256),
        pad_id=tokenizer.pad_id,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    history: list[dict[str, float]] = []
    best_val_loss = 1e9
    best_epoch = -1
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tok = 0
        for xb, yb in make_batches(x_tr, y_tr, batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed + epoch):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(yb.numel())
            total_tok += int(yb.numel())
        train_loss = total_loss / max(total_tok, 1)
        val_loss, val_acc = evaluate(model, x_val, y_val, batch_size=cfg.batch_size, loss_fn=loss_fn, device=device)
        perplexity = float(np.exp(min(val_loss, 20)))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "val_ppl": float(perplexity),
            }
        )
        print(
            f"[Epoch {epoch:02d}/{cfg.epochs}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_ppl={perplexity:.2f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))
    sample = sample_text(model, tokenizer, prefix="this ", max_new_tokens=120, device=device)

    (run_dir / "sample.txt").write_text(sample, encoding="utf-8")
    (run_dir / "tokenizer.json").write_text(json.dumps({"itos": tokenizer.itos}, ensure_ascii=False), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_ppl": float(np.exp(min(best_val_loss, 20))),
                "num_train_seq": int(len(x_tr)),
                "num_val_seq": int(len(x_val)),
                "vocab_size": tokenizer.vocab_size,
                "device_used": str(device),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "val_ppl"])
        writer.writeheader()
        writer.writerows(history)

    mpl_cache = run_dir / ".mplconfig"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    font_cache = run_dir / ".cache"
    font_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(font_cache))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([h["epoch"] for h in history], [h["val_loss"] for h in history], label="val_loss")
    ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="train_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "curve.png", dpi=150)
    plt.close(fig)
    return run_dir


def parse_args() -> TrainConfig:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train Task-3 decoder-only LM")
    parser.add_argument("--corpus-path", type=str, default=str(project_root / "data/task3/lm_corpus.txt"))
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default="outputs/task3")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    return TrainConfig(
        corpus_path=args.corpus_path,
        seq_len=args.seq_len,
        stride=args.stride,
        seed=args.seed,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_ratio=args.val_ratio,
        device=args.device,
        save_dir=args.save_dir,
        run_name=args.run_name,
    )


def main() -> None:
    cfg = parse_args()
    out = run_training(cfg)
    print(f"Saved artifacts to: {out}")


if __name__ == "__main__":
    main()

