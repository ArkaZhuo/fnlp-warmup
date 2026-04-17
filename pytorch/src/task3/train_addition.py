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

from task3.models import Seq2SeqTransformer
from task3.tokenizer import CharTokenizer


@dataclass
class TrainConfig:
    train_tsv: str
    test_tsv: str
    seed: int
    max_src_len: int
    max_tgt_len: int
    d_model: int
    nhead: int
    enc_layers: int
    dec_layers: int
    ff_dim: int
    dropout: float
    batch_size: int
    epochs: int
    lr: float
    device: str
    save_dir: str
    run_name: str | None
    reverse_src: bool
    reverse_tgt: bool
    init_checkpoint: str | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_addition(path: str | Path) -> tuple[list[str], list[str]]:
    df = pd.read_csv(path, sep="\t")
    return df["src"].astype(str).tolist(), df["tgt"].astype(str).tolist()


def reverse_expr(expr: str) -> str:
    if "+" not in expr:
        return expr[::-1]
    a, b = expr.split("+", 1)
    return f"{a[::-1]}+{b[::-1]}"


def encode_samples(
    src_texts: list[str],
    tgt_texts: list[str],
    tokenizer: CharTokenizer,
    max_src_len: int,
    max_tgt_len: int,
    reverse_src: bool = False,
    reverse_tgt: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    src_arr = np.full((len(src_texts), max_src_len), tokenizer.pad_id, dtype=np.int64)
    tgt_arr = np.full((len(tgt_texts), max_tgt_len), tokenizer.pad_id, dtype=np.int64)
    for i, (s, t) in enumerate(zip(src_texts, tgt_texts)):
        if reverse_src:
            s = reverse_expr(s)
        if reverse_tgt:
            t = t[::-1]
        src_ids = tokenizer.encode(s, add_bos=False, add_eos=True)[:max_src_len]
        tgt_ids = tokenizer.encode(t, add_bos=True, add_eos=True)[:max_tgt_len]
        src_arr[i, : len(src_ids)] = np.asarray(src_ids, dtype=np.int64)
        tgt_arr[i, : len(tgt_ids)] = np.asarray(tgt_ids, dtype=np.int64)
    return src_arr, tgt_arr


def make_batches(src: np.ndarray, tgt: np.ndarray, batch_size: int, seed: int, shuffle: bool) -> list[tuple[torch.Tensor, torch.Tensor]]:
    idx = np.arange(len(src))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(0, len(idx), batch_size):
        b = idx[i : i + batch_size]
        batches.append((torch.tensor(src[b], dtype=torch.long), torch.tensor(tgt[b], dtype=torch.long)))
    return batches


@torch.no_grad()
def greedy_decode(
    model: Seq2SeqTransformer,
    src: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int,
) -> torch.Tensor:
    model.eval()
    batch = src.size(0)
    out = torch.full((batch, 1), bos_id, dtype=torch.long, device=src.device)
    for _ in range(max_len - 1):
        logits = model(src, out)
        nxt = logits[:, -1, :].argmax(dim=1, keepdim=True)
        out = torch.cat([out, nxt], dim=1)
        if (nxt.squeeze(1) == eos_id).all():
            break
    return out


def sequence_exact_match(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    eos_id: int,
    pad_id: int,
) -> float:
    # compare decoded tokens after bos until eos/pad
    def normalize(arr: torch.Tensor) -> list[list[int]]:
        rows: list[list[int]] = []
        for row in arr.tolist():
            seq: list[int] = []
            for x in row:
                if x == pad_id:
                    continue
                if x == eos_id:
                    break
                seq.append(x)
            rows.append(seq)
        return rows

    p = normalize(pred[:, 1:])  # drop bos
    g = normalize(tgt[:, 1:])  # drop bos
    correct = sum(int(a == b) for a, b in zip(p, g))
    return correct / max(len(g), 1)


def run_training(cfg: TrainConfig) -> Path:
    set_seed(cfg.seed)
    project_root = Path(__file__).resolve().parents[2]
    save_root = Path(cfg.save_dir)
    if not save_root.is_absolute():
        save_root = project_root / save_root
    run_name = cfg.run_name or f"add_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = save_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_src, train_tgt = load_addition(cfg.train_tsv)
    test_src, test_tgt = load_addition(cfg.test_tsv)
    corpus_src = [reverse_expr(s) if cfg.reverse_src else s for s in (train_src + test_src)]
    corpus_tgt = [t[::-1] if cfg.reverse_tgt else t for t in (train_tgt + test_tgt)]
    corpus = corpus_src + corpus_tgt + ["+"]
    tokenizer = CharTokenizer.build(corpus)
    x_train, y_train = encode_samples(
        train_src,
        train_tgt,
        tokenizer,
        cfg.max_src_len,
        cfg.max_tgt_len,
        reverse_src=cfg.reverse_src,
        reverse_tgt=cfg.reverse_tgt,
    )
    x_test, y_test = encode_samples(
        test_src,
        test_tgt,
        tokenizer,
        cfg.max_src_len,
        cfg.max_tgt_len,
        reverse_src=cfg.reverse_src,
        reverse_tgt=cfg.reverse_tgt,
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = Seq2SeqTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.enc_layers,
        num_decoder_layers=cfg.dec_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        max_len=max(cfg.max_src_len, cfg.max_tgt_len) + 8,
        pad_id=tokenizer.pad_id,
    ).to(device)
    if cfg.init_checkpoint:
        ckpt_path = Path(cfg.init_checkpoint)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=False)
            print(f"[Init] loaded checkpoint: {ckpt_path}")
        else:
            print(f"[Init] checkpoint not found, skip: {ckpt_path}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    history: list[dict[str, float]] = []
    best_test_exact = -1.0
    best_epoch = -1
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tok = 0
        for xb, yb in make_batches(x_train, y_train, cfg.batch_size, seed=cfg.seed + epoch, shuffle=True):
            xb = xb.to(device)
            yb = yb.to(device)
            tgt_in = yb[:, :-1]
            tgt_out = yb[:, 1:]
            optimizer.zero_grad()
            logits = model(xb, tgt_in)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            valid = int((tgt_out != tokenizer.pad_id).sum().item())
            total_loss += float(loss.item()) * max(valid, 1)
            total_tok += max(valid, 1)

        # eval exact match on test set
        model.eval()
        with torch.no_grad():
            exact_sum = 0.0
            exact_count = 0
            for xb, yb in make_batches(x_test, y_test, cfg.batch_size, seed=0, shuffle=False):
                xb = xb.to(device)
                yb = yb.to(device)
                out = greedy_decode(
                    model=model,
                    src=xb,
                    bos_id=tokenizer.bos_id,
                    eos_id=tokenizer.eos_id,
                    max_len=cfg.max_tgt_len,
                )
                batch_exact = sequence_exact_match(
                    out,
                    yb,
                    eos_id=tokenizer.eos_id,
                    pad_id=tokenizer.pad_id,
                )
                exact_sum += batch_exact * xb.size(0)
                exact_count += xb.size(0)
            test_exact = exact_sum / max(exact_count, 1)

        train_loss = total_loss / max(total_tok, 1)
        history.append({"epoch": float(epoch), "train_loss": float(train_loss), "test_exact": float(test_exact)})
        print(f"[Epoch {epoch:02d}/{cfg.epochs}] train_loss={train_loss:.4f} test_exact={test_exact:.4f}")

        if test_exact > best_test_exact:
            best_test_exact = test_exact
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    # save artifacts
    (run_dir / "tokenizer.json").write_text(json.dumps({"itos": tokenizer.itos}, ensure_ascii=False), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_test_exact": best_test_exact,
                "num_train": len(train_src),
                "num_test": len(test_src),
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

    # quick figure
    mpl_cache = run_dir / ".mplconfig"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    font_cache = run_dir / ".cache"
    font_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(font_cache))
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="train_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(
        [h["epoch"] for h in history],
        [h["test_exact"] for h in history],
        color="tab:orange",
        label="test_exact",
    )
    ax2.set_ylabel("Exact Match")
    fig.tight_layout()
    fig.savefig(run_dir / "curve.png", dpi=150)
    plt.close(fig)
    return run_dir


def parse_args() -> TrainConfig:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train Task-3 addition transformer")
    parser.add_argument("--train-tsv", type=str, default=str(project_root / "data/task3/add_train.tsv"))
    parser.add_argument("--test-tsv", type=str, default=str(project_root / "data/task3/add_test.tsv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-src-len", type=int, default=16)
    parser.add_argument("--max-tgt-len", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--enc-layers", type=int, default=2)
    parser.add_argument("--dec-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default="outputs/task3")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--reverse-src", action="store_true", default=False)
    parser.add_argument("--reverse-tgt", action="store_true", default=False)
    parser.add_argument("--init-checkpoint", type=str, default=None)
    args = parser.parse_args()
    return TrainConfig(
        train_tsv=args.train_tsv,
        test_tsv=args.test_tsv,
        seed=args.seed,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        d_model=args.d_model,
        nhead=args.nhead,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        run_name=args.run_name,
        reverse_src=args.reverse_src,
        reverse_tgt=args.reverse_tgt,
        init_checkpoint=args.init_checkpoint,
    )


def main() -> None:
    cfg = parse_args()
    out = run_training(cfg)
    print(f"Saved artifacts to: {out}")


if __name__ == "__main__":
    main()
