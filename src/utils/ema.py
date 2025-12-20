# src/utils/ema.py
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn


def init_ema_model(student_model: nn.Module, device: Optional[torch.device] = None) -> nn.Module:
    """
    Create EMA teacher as a deepcopy of student.
    - Teacher starts with identical weights
    - Set to eval mode by default
    - Grad disabled for teacher parameters (recommended)
    """
    ema_model = copy.deepcopy(student_model)

    if device is None:
        try:
            device = next(student_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    ema_model.to(device)
    ema_model.eval()

    for p in ema_model.parameters():
        p.requires_grad_(False)

    return ema_model


@torch.no_grad()
def update_ema_model(ema_model: nn.Module, student_model: nn.Module, alpha: float = 0.99) -> None:
    """
    EMA update: θ_te <- α θ_te + (1-α) θ

    Notes:
    - Assumes ema_model and student_model have identical parameter structure.
    - Uses in-place ops under no_grad.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    for p_te, p in zip(ema_model.parameters(), student_model.parameters()):
        p_te.data.mul_(alpha).add_(p.data, alpha=1.0 - alpha)
