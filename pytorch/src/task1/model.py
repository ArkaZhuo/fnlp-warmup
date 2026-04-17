from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class BatchMetrics:
    loss: float
    correct: int
    total: int


class LinearClassifierScratch:
    """
    Multi-class linear classifier trained with manual gradients.
    No torch.nn modules are used.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        device: torch.device,
        seed: int = 42,
    ) -> None:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        weight = torch.randn(input_dim, num_classes, generator=g, dtype=torch.float32)
        weight = weight * 0.01
        bias = torch.zeros(num_classes, dtype=torch.float32)
        self.W = weight.to(device)
        self.b = bias.to(device)
        self.device = device
        self.input_dim = input_dim
        self.num_classes = num_classes

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W + self.b

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x).argmax(dim=1)

    def _cross_entropy(self, logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = logits.shape[0]
        max_logits = logits.max(dim=1, keepdim=True).values
        shifted = logits - max_logits
        exp_shifted = torch.exp(shifted)
        probs = exp_shifted / exp_shifted.sum(dim=1, keepdim=True)
        losses = -torch.log(probs[torch.arange(batch_size, device=logits.device), y] + 1e-12)
        loss = losses.mean()

        dlogits = probs
        dlogits[torch.arange(batch_size, device=logits.device), y] -= 1.0
        dlogits = dlogits / batch_size
        return loss, dlogits

    def _mse(self, logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        targets = torch.zeros_like(logits)
        targets[torch.arange(logits.shape[0], device=logits.device), y] = 1.0
        diff = logits - targets
        loss = (diff * diff).mean()
        dlogits = 2.0 * diff / diff.numel()
        return loss, dlogits

    def train_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lr: float,
        loss_name: str = "ce",
        weight_decay: float = 0.0,
    ) -> BatchMetrics:
        logits = self.logits(x)
        if loss_name == "ce":
            loss_tensor, dlogits = self._cross_entropy(logits, y)
        elif loss_name == "mse":
            loss_tensor, dlogits = self._mse(logits, y)
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")

        dW = x.T @ dlogits
        if weight_decay > 0:
            dW = dW + weight_decay * self.W
        db = dlogits.sum(dim=0)

        self.W -= lr * dW
        self.b -= lr * db

        pred = logits.argmax(dim=1)
        correct = int((pred == y).sum().item())
        total = int(y.numel())
        return BatchMetrics(loss=float(loss_tensor.item()), correct=correct, total=total)

    def eval_batch(self, x: torch.Tensor, y: torch.Tensor, loss_name: str = "ce") -> BatchMetrics:
        logits = self.logits(x)
        if loss_name == "ce":
            loss_tensor, _ = self._cross_entropy(logits, y)
        elif loss_name == "mse":
            loss_tensor, _ = self._mse(logits, y)
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")
        pred = logits.argmax(dim=1)
        correct = int((pred == y).sum().item())
        total = int(y.numel())
        return BatchMetrics(loss=float(loss_tensor.item()), correct=correct, total=total)

    def save(self, path: str) -> None:
        torch.save({"W": self.W.cpu(), "b": self.b.cpu()}, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.W = state["W"].to(self.device)
        self.b = state["b"].to(self.device)


def sparse_batch_to_dense(
    sparse_samples: Sequence[dict[int, float]],
    input_dim: int,
    device: torch.device,
    l1_normalize: bool = True,
) -> torch.Tensor:
    """
    Convert sparse count dict list into dense [B, input_dim] matrix.
    """
    batch_size = len(sparse_samples)
    x = torch.zeros((batch_size, input_dim), dtype=torch.float32, device=device)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for row_id, sample in enumerate(sparse_samples):
        for col_id, value in sample.items():
            if 0 <= col_id < input_dim:
                rows.append(row_id)
                cols.append(col_id)
                vals.append(float(value))

    if rows:
        row_idx = torch.tensor(rows, dtype=torch.long, device=device)
        col_idx = torch.tensor(cols, dtype=torch.long, device=device)
        val_tensor = torch.tensor(vals, dtype=torch.float32, device=device)
        x.index_put_((row_idx, col_idx), val_tensor, accumulate=True)

    if l1_normalize:
        row_sum = x.sum(dim=1, keepdim=True).clamp_min(1e-8)
        x = x / row_sum
    return x
