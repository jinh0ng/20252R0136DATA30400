# src/utils/graph.py
from __future__ import annotations

from typing import Iterable, List, Tuple, Optional

import numpy as np


def build_dense_adj(
    num_classes: int,
    edges: List[Tuple[int, int]],
    undirected: bool = True,
    self_loop: bool = True,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Build a dense adjacency matrix (C, C) with 0/1 entries.
    This matches your notebook usage (undirected + self-loop).

    Returns:
        adj: np.ndarray shape (C, C), dtype float32 by default
    """
    C = num_classes
    adj = np.zeros((C, C), dtype=dtype)

    for p, c in edges:
        if 0 <= p < C and 0 <= c < C:
            adj[p, c] = 1.0
            if undirected:
                adj[c, p] = 1.0

    if self_loop:
        np.fill_diagonal(adj, 1.0)

    return adj


def to_torch_adj(adj: np.ndarray, device: Optional[str] = None):
    """
    Convert numpy adjacency to torch.FloatTensor on device.
    """
    try:
        import torch
    except Exception as e:
        raise RuntimeError("torch is required for to_torch_adj()") from e

    t = torch.from_numpy(adj).float()
    if device is not None:
        t = t.to(device)
    return t
