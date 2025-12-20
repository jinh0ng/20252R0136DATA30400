# src/utils/datasets.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """
    Dataset for (doc_embedding, multi-hot labels).
    X: (N, d), y: (N, C)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X rows ({X.shape[0]}) != y rows ({y.shape[0]})")
        if X.ndim != 2 or y.ndim != 2:
            raise ValueError(f"Expected X,y to be 2D arrays, got X.ndim={X.ndim}, y.ndim={y.ndim}")

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
