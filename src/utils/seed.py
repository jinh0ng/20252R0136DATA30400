# src/utils/seed.py
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.
    - deterministic=True sets cudnn deterministic flags when torch is available.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """
    For DataLoader worker_init_fn to make numpy/random deterministic per-worker.
    """
    # torch initial seed is set in DataLoader; use it to derive numpy/random seeds
    if torch is None:
        base_seed = 42
    else:
        base_seed = torch.initial_seed() % 2**32  # type: ignore[attr-defined]
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def torch_generator(seed: int = 42):
    """
    Convenience: generator for random_split / dataloader shuffling.
    """
    if torch is None:
        return None
    g = torch.Generator()
    g.manual_seed(seed)
    return g
