# src/utils/taxonomy.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Taxonomy:
    """
    Simple taxonomy utility built from edges (parent, child).
    Assumes a single parent per node (tree/forest), as per your notebook logic.
    """
    num_classes: int
    edges: List[Tuple[int, int]]

    def __post_init__(self) -> None:
        self.children: Dict[int, List[int]] = defaultdict(list)
        self.parent: Dict[int, Optional[int]] = {cid: None for cid in range(self.num_classes)}

        for p, c in self.edges:
            if 0 <= p < self.num_classes and 0 <= c < self.num_classes:
                self.children[p].append(c)
                self.parent[c] = p

        self.roots: List[int] = [cid for cid, p in self.parent.items() if p is None]

        # Cache for paths/ancestors (lazy fill)
        self._anc_cache: Dict[int, List[int]] = {}
        self._path_cache: Dict[int, List[int]] = {}

    def get_ancestors(self, cid: int) -> List[int]:
        """
        Ancestors of cid (excluding itself), ordered from root -> ... -> parent(cid).
        """
        if cid in self._anc_cache:
            return self._anc_cache[cid]

        res: List[int] = []
        cur = self.parent.get(cid, None)
        while cur is not None:
            res.append(cur)
            cur = self.parent.get(cur, None)
        res = res[::-1]
        self._anc_cache[cid] = res
        return res

    def get_path(self, cid: int) -> List[int]:
        """
        Path root -> ... -> cid (including itself).
        """
        if cid in self._path_cache:
            return self._path_cache[cid]
        path = self.get_ancestors(cid) + [cid]
        self._path_cache[cid] = path
        return path

    def get_descendants(self, cid: int) -> List[int]:
        """
        Returns all descendants of cid (excluding itself).
        """
        res: List[int] = []
        stack = [cid]
        while stack:
            cur = stack.pop()
            for ch in self.children.get(cur, []):
                res.append(ch)
                stack.append(ch)
        return res

    def build_all_paths(self) -> Dict[int, List[int]]:
        """
        Convenience: materialize path cache for all classes.
        """
        return {cid: self.get_path(cid) for cid in range(self.num_classes)}
