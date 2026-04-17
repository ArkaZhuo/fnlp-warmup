from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_len: int = 64,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device), diagonal=1).bool()

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_pad = src.eq(self.pad_id)
        tgt_pad = tgt_in.eq(self.pad_id)
        tgt_mask = self._causal_mask(tgt_in.size(1), tgt_in.device)

        src_h = self.pos(self.src_emb(src) * math.sqrt(self.d_model))
        tgt_h = self.pos(self.tgt_emb(tgt_in) * math.sqrt(self.d_model))
        out = self.transformer(
            src_h,
            tgt_h,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
            tgt_mask=tgt_mask,
        )
        return self.out(out)


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_len: int = 256,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device), diagonal=1).bool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_mask = x.eq(self.pad_id)
        causal = self._causal_mask(x.size(1), x.device)
        h = self.embedding(x) * math.sqrt(self.d_model)
        h = self.pos(h)
        h = self.encoder(h, mask=causal, src_key_padding_mask=pad_mask)
        return self.out(h)

