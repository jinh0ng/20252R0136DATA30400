# src/utils/coreclass.py
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .taxonomy import Taxonomy


def compute_core_classes_path_mean(
    train_doc_emb: np.ndarray,
    class_emb: np.ndarray,
    taxonomy: Taxonomy,
    top_k_core: int = 3,
    min_score: Optional[float] = None,
    batch_size: int = 1024,
    verbose: bool = True,
) -> List[List[int]]:
    """
    logic:
      sim = doc_emb @ class_emb.T
      path_score(c) = mean(sim over nodes in path(root->c))
      core = top_k_core classes by path_score

    This implementation computes in batches to control peak memory.

    Returns:
        core_classes_per_doc: List[List[int]] of length N_train
    """
    N, dim = train_doc_emb.shape
    C, dim2 = class_emb.shape
    if dim != dim2:
        raise ValueError(f"Dimension mismatch: doc_dim={dim}, class_dim={dim2}")
    if C != taxonomy.num_classes:
        raise ValueError(f"num_classes mismatch: class_emb has C={C}, taxonomy={taxonomy.num_classes}")

    # Precompute paths for all classes (list of index lists)
    paths: Dict[int, List[int]] = taxonomy.build_all_paths()

    core_classes_per_doc: List[List[int]] = []
    if verbose:
        print(f"[coreclass] Computing core classes with batch_size={batch_size} ...")

    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        docs = train_doc_emb[start:end]  # (B, d)

        # (B, C) dot product (assumes embeddings are normalized -> cosine)
        sim = docs @ class_emb.T

        # Compute path_scores (B, C) by looping classes (C=531 is manageable)
        B = sim.shape[0]
        path_scores = np.empty((B, C), dtype=np.float32)

        for cid in range(C):
            idx = paths[cid]
            # mean over path nodes
            path_scores[:, cid] = sim[:, idx].mean(axis=1)

        # Select top-k per row
        for i in range(B):
            row = path_scores[i]
            idx_sorted = np.argsort(-row)  # descending
            if min_score is not None:
                idx_sorted = [cid for cid in idx_sorted if row[cid] >= min_score]
                if len(idx_sorted) < top_k_core:
                    # fallback: ensure at least top_k_core exist
                    idx_sorted = np.argsort(-row).tolist()
            core = idx_sorted[:top_k_core]
            core_classes_per_doc.append(list(map(int, core)))

        if verbose and (start == 0 or end == N or (start // batch_size) % 10 == 0):
            print(f"[coreclass] processed {end}/{N}")

    return core_classes_per_doc
