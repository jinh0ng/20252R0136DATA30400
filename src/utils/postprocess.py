# src/utils/postprocess.py
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def dynamic_topk_labels(
    probs: np.ndarray,
    threshold: float = 0.65,
    min_k: int = 2,
    max_k: int = 3,
    sort_labels: bool = True,
) -> List[List[int]]:
    """
    dynamic top-k rule per row of probs.

    Args:
        probs: (N, C) probabilities
        threshold: select candidates where prob >= threshold
        min_k: if fewer than min_k candidates, force top min_k
        max_k: if more than max_k candidates, keep top max_k among candidates
        sort_labels: if True, sort final label ids ascending (leaderboard stability)

    Returns:
        labels: list of list[int] length N
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D (N,C), got shape {probs.shape}")

    N, C = probs.shape
    out: List[List[int]] = []

    for i in range(N):
        row = probs[i]

        candidates = np.where(row >= threshold)[0].tolist()

        if len(candidates) < min_k:
            topk = row.argsort()[-min_k:][::-1]
            candidates = topk.tolist()
        elif len(candidates) > max_k:
            # pick top max_k among the candidates by prob
            cand_probs = row[candidates]
            top_idx = np.argsort(cand_probs)[-max_k:][::-1]
            candidates = [candidates[j] for j in top_idx]

        if sort_labels:
            candidates = sorted(candidates)

        out.append([int(x) for x in candidates])

    return out


def labels_to_csv_strings(labels: List[List[int]]) -> List[str]:
    """
    Convert label id lists to 'a,b,c' strings.
    """
    return [",".join(str(x) for x in row) for row in labels]
