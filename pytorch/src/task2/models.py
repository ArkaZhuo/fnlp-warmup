from __future__ import annotations

import math

import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        num_kernels: int = 64,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_kernels, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_kernels * len(kernel_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L]
        emb = self.embedding(x)  # [B, L, E]
        emb = emb.transpose(1, 2)  # [B, E, L]
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(emb))
            p = torch.max(h, dim=2).values
            pooled.append(p)
        feats = torch.cat(pooled, dim=1)
        feats = self.dropout(feats)
        return self.fc(feats)


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        pooled = out.max(dim=1).values
        pooled = self.dropout(pooled)
        return self.fc(pooled)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.2,
        max_len: int = 512,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=max_len, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L]
        pad_mask = x.eq(self.pad_idx)
        emb = self.embedding(x)
        emb = self.pos_encoding(emb)
        h = self.encoder(emb, src_key_padding_mask=pad_mask)
        # mean pooling ignoring pad positions
        valid = (~pad_mask).unsqueeze(-1).float()
        h_sum = (h * valid).sum(dim=1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        pooled = h_sum / denom
        pooled = self.dropout(pooled)
        return self.fc(pooled)


def build_model(
    model_name: str,
    vocab_size: int,
    num_classes: int,
    embed_dim: int,
    pad_idx: int,
    max_len: int,
    num_kernels: int = 64,
    kernel_sizes: tuple[int, ...] = (3, 4, 5),
    hidden_dim: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    ff_dim: int = 256,
    dropout: float = 0.3,
) -> nn.Module:
    if model_name == "cnn":
        return TextCNN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_classes=num_classes,
            kernel_sizes=kernel_sizes,
            num_kernels=num_kernels,
            dropout=dropout,
            pad_idx=pad_idx,
        )
    if model_name == "rnn":
        return BiLSTMClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=1,
            dropout=dropout,
            pad_idx=pad_idx,
        )
    if model_name == "transformer":
        return TransformerClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_classes=num_classes,
            nhead=nhead,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            max_len=max_len,
            pad_idx=pad_idx,
        )
    raise ValueError(f"Unknown model_name: {model_name}")

