# src/utils/train.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .ema import update_ema_model


@dataclass
class EpochStats:
    loss: float
    num_samples: int


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    ema_model: Optional[nn.Module] = None,
    alpha_ema: Optional[float] = None,
) -> EpochStats:
    """
    Train model for one epoch.
    If ema_model and alpha_ema are provided, update EMA after each optimizer step.
    """
    model.train()
    total_loss = 0.0
    total_n = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        if ema_model is not None and alpha_ema is not None:
            update_ema_model(ema_model, model, alpha=alpha_ema)

        bs = batch_x.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    avg_loss = total_loss / max(total_n, 1)
    return EpochStats(loss=avg_loss, num_samples=total_n)


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochStats:
    """
    Evaluate model for one epoch.
    """
    model.eval()
    total_loss = 0.0
    total_n = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        bs = batch_x.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    avg_loss = total_loss / max(total_n, 1)
    return EpochStats(loss=avg_loss, num_samples=total_n)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    ema_model: Optional[nn.Module] = None,
    alpha_ema: Optional[float] = None,
) -> Dict[str, list]:
    """
    Simple fit loop returning history dict.
    """
    history = {"train_loss": [], "valid_loss": []}

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            ema_model=ema_model,
            alpha_ema=alpha_ema,
        )
        va = eval_one_epoch(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
        )
        history["train_loss"].append(tr.loss)
        history["valid_loss"].append(va.loss)

        print(f"[Epoch {epoch:02d}] train={tr.loss:.4f} | valid={va.loss:.4f}")

    return history
