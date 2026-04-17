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
import pandas as pd
import torch
import torch.nn as nn

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from task3.models import DecoderOnlyTransformer
from task3.tokenizer import CharTokenizer


@dataclass
class TrainConfig:
    train_tsv: str
    test_tsv: str
    seed: int
    seq_len: int
    d_model: int
    nhead: int
    num_layers: int
    ff_dim: int
    dropout: float
    batch_size: int
    epochs: int
    lr: float
    device: str
    save_dir: str
    run_name: str | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_pairs(path: str | Path) -> list[tuple[str, str]]:
    df = pd.read_csv(path, sep="\t")
    return list(zip(df["src"].astype(str).tolist(), df["tgt"].astype(str).tolist()))


def build_sequences(
    pairs: list[tuple[str, str]],
    tokenizer: CharTokenizer,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.full((len(pairs), seq_len), tokenizer.pad_id, dtype=np.int64)
    y = np.full((len(pairs), seq_len), tokenizer.pad_id, dtype=np.int64)
    sep_id = tokenizer.stoi["="]
    mask = np.zeros((len(pairs), seq_len), dtype=np.int64)

    for i, (src, tgt) in enumerate(pairs):
        full = f"{src}={tgt}"
        ids = tokenizer.encode(full, add_bos=True, add_eos=True)
        ids = ids[: seq_len + 1]
        if len(ids) < 2:
            continue
        inp = ids[:-1]
        out = ids[1:]
        n = min(len(inp), seq_len)
        x[i, :n] = np.asarray(inp[:n], dtype=np.int64)
        y[i, :n] = np.asarray(out[:n], dtype=np.int64)

        # only supervise/evaluate target side tokens after '='
        sep_pos = -1
        for j in range(n):
            if x[i, j] == sep_id:
                sep_pos = j
                break
        if sep_pos >= 0:
            mask[i, sep_pos:] = 1
    return x, y, mask


def make_batches(x: np.ndarray, y: np.ndarray, m: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    idx = np.arange(len(x))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    out = []
    for i in range(0, len(idx), batch_size):
        b = idx[i : i + batch_size]
        out.append((x[b], y[b], m[b]))
    return out


@torch.no_grad()
def eval_exact(
    model: DecoderOnlyTransformer,
    pairs: list[tuple[str, str]],
    tokenizer: CharTokenizer,
    seq_len: int,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    for src, tgt in pairs:
        prompt = f"{src}="
        ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        for _ in range(len(tgt) + 2):
            if x.size(1) >= seq_len:
                break
            logits = model(x)
            nxt = int(logits[0, -1].argmax().item())
            x = torch.cat([x, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
            if nxt == tokenizer.eos_id:
                break
        pred = tokenizer.decode(x[0].tolist(), skip_special=True)
        if "=" in pred:
            pred_ans = pred.split("=", 1)[1].strip()
        else:
            pred_ans = pred.strip()
        pred_ans = pred_ans.split("\n")[0]
        if pred_ans == tgt:
            correct += 1
    return correct / max(len(pairs), 1)


def run_training(cfg: TrainConfig) -> Path:
    set_seed(cfg.seed)
    project_root = Path(__file__).resolve().parents[2]
    save_root = Path(cfg.save_dir)
    if not save_root.is_absolute():
        save_root = project_root / save_root
    run_name = cfg.run_name or f"add_decoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = save_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_pairs = load_pairs(cfg.train_tsv)
    test_pairs = load_pairs(cfg.test_tsv)
    corpus = [f"{s}={t}" for s, t in (train_pairs + test_pairs)] + ["="]
    tokenizer = CharTokenizer.build(corpus)
    x_tr, y_tr, m_tr = build_sequences(train_pairs, tokenizer, seq_len=cfg.seq_len)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        max_len=cfg.seq_len + 16,
        pad_id=tokenizer.pad_id,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ce = nn.CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_id)

    history = []
    best_exact = -1.0
    best_epoch = -1
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tok = 0
        for xb, yb, mb in make_batches(x_tr, y_tr, m_tr, cfg.batch_size, shuffle=True, seed=cfg.seed + epoch):
            x_t = torch.tensor(xb, dtype=torch.long, device=device)
            y_t = torch.tensor(yb, dtype=torch.long, device=device)
            m_t = torch.tensor(mb, dtype=torch.float32, device=device)
            optimizer.zero_grad()
            logits = model(x_t)
            per_tok = ce(logits.reshape(-1, logits.size(-1)), y_t.reshape(-1)).reshape(y_t.size())
            loss = (per_tok * m_t).sum() / m_t.sum().clamp_min(1.0)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * float(m_t.sum().item())
            total_tok += int(m_t.sum().item())
        train_loss = total_loss / max(total_tok, 1)
        test_exact = eval_exact(model, test_pairs, tokenizer, seq_len=cfg.seq_len, device=device)
        history.append({"epoch": float(epoch), "train_loss": float(train_loss), "test_exact": float(test_exact)})
        print(f"[Epoch {epoch:02d}/{cfg.epochs}] train_loss={train_loss:.4f} test_exact={test_exact:.4f}")
        if test_exact > best_exact:
            best_exact = test_exact
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    (run_dir / "tokenizer.json").write_text(json.dumps({"itos": tokenizer.itos}, ensure_ascii=False), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_test_exact": best_exact,
                "num_train": len(train_pairs),
                "num_test": len(test_pairs),
                "vocab_size": tokenizer.vocab_size,
                "device_used": str(device),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "test_exact"])
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
    ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="train_loss")
    ax2 = ax.twinx()
    ax2.plot([h["epoch"] for h in history], [h["test_exact"] for h in history], color="tab:orange", label="test_exact")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("Exact")
    fig.tight_layout()
    fig.savefig(run_dir / "curve.png", dpi=150)
    plt.close(fig)
    return run_dir


def parse_args() -> TrainConfig:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Task-3 addition decoder-only variant")
    parser.add_argument("--train-tsv", type=str, default=str(project_root / "data/task3/add_train.tsv"))
    parser.add_argument("--test-tsv", type=str, default=str(project_root / "data/task3/add_test.tsv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default="outputs/task3")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    return TrainConfig(
        train_tsv=args.train_tsv,
        test_tsv=args.test_tsv,
        seed=args.seed,
        seq_len=args.seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
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

