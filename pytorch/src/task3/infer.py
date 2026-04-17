from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from task3.models import Seq2SeqTransformer
from task3.tokenizer import CharTokenizer
from task3.train_addition import greedy_decode, reverse_expr


def build_tokenizer_from_file(path: Path) -> CharTokenizer:
    payload = json.loads(path.read_text(encoding="utf-8"))
    itos = payload["itos"]
    stoi = {t: i for i, t in enumerate(itos)}
    return CharTokenizer(stoi=stoi, itos=itos)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for Task-3 addition model")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--expr", type=str, required=True, help="Expression like 123+456")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer = build_tokenizer_from_file(run_dir / "tokenizer.json")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = Seq2SeqTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_encoder_layers=int(cfg["enc_layers"]),
        num_decoder_layers=int(cfg["dec_layers"]),
        ff_dim=int(cfg["ff_dim"]),
        dropout=float(cfg["dropout"]),
        max_len=max(int(cfg["max_src_len"]), int(cfg["max_tgt_len"])) + 8,
        pad_id=tokenizer.pad_id,
    ).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True))
    if bool(cfg.get("reverse_src", False)):
        src_text = reverse_expr(args.expr)
    else:
        src_text = args.expr
    src_ids = tokenizer.encode(src_text, add_bos=False, add_eos=True)
    src = torch.full((1, int(cfg["max_src_len"])), tokenizer.pad_id, dtype=torch.long, device=device)
    src[0, : min(len(src_ids), src.size(1))] = torch.tensor(src_ids[: src.size(1)], device=device)
    out = greedy_decode(model, src, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id, max_len=int(cfg["max_tgt_len"]))
    pred = tokenizer.decode(out[0].tolist(), skip_special=True)
    if bool(cfg.get("reverse_tgt", False)):
        pred = pred[::-1]
    print(f"expr={args.expr}")
    print(f"pred={pred}")


if __name__ == "__main__":
    main()
