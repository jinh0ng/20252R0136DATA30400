# src/utils/pseudo_label.py
from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn


def _infer_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def refine_labels_with_ema_teacher(
    ema_model: nn.Module,
    X_np: np.ndarray,
    Y_old: np.ndarray,
    t_pos: float = 0.9,
    t_neg: float = 0.1,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    EMA teacher로 pseudo-label 정제:

    - prob >= t_pos  -> 1로 강제 (확실한 positive)
    - prob <= t_neg  -> (기존이 1인 경우) 0으로 (확실한 negative)

    Args:
        ema_model: teacher model (eval recommended)
        X_np: (N, d) numpy
        Y_old: (N, C) numpy multi-hot (0/1)
        t_pos/t_neg: thresholds
        device: if None, inferred from ema_model
    Returns:
        Y_new: refined labels (N, C)
        probs: teacher probs (N, C)
    """
    if device is None:
        device = _infer_device(ema_model)

    ema_model.eval()

    X = torch.from_numpy(X_np).float().to(device)
    logits = ema_model(X)                               # (N, C)
    probs = torch.sigmoid(logits).detach().cpu().numpy()

    Y_new = Y_old.copy()

    pos_mask = probs >= t_pos
    Y_new[pos_mask] = 1.0

    neg_mask = probs <= t_neg
    drop_mask = (Y_new == 1.0) & neg_mask
    Y_new[drop_mask] = 0.0

    if verbose:
        print("Refinement summary (EMA):")
        print("  newly set to 1:", int(pos_mask.sum()))
        print("  dropped from 1:", int(drop_mask.sum()))

    return Y_new, probs
