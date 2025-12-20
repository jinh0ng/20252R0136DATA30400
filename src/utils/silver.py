# src/utils/silver.py
from __future__ import annotations

from typing import List, Sequence, Set

import numpy as np

from .taxonomy import Taxonomy


def build_taxo_silver_labels(
    core_classes_per_doc: List[List[int]],
    taxonomy: Taxonomy,
    num_classes: int,
    include_ancestors: bool = True,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    logic:
      C_pos(i) = core classes ∪ ancestors(core)
      Y[i, cid] = 1 for cid in C_pos(i)

    Returns:
      Y: (N, C) float32
    """
    N = len(core_classes_per_doc)
    C = num_classes
    if C != taxonomy.num_classes:
        raise ValueError(f"num_classes mismatch: num_classes={C}, taxonomy={taxonomy.num_classes}")

    Y = np.zeros((N, C), dtype=dtype)

    for i, core_list in enumerate(core_classes_per_doc):
        pos: Set[int] = set()
        for cid in core_list:
            pos.add(int(cid))
            if include_ancestors:
                for anc in taxonomy.get_ancestors(int(cid)):
                    pos.add(int(anc))
        for cid in pos:
            if 0 <= cid < C:
                Y[i, cid] = 1.0

    return Y
