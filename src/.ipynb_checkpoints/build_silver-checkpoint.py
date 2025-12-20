# src/build_silver.py
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

# Ensure src/ is on sys.path so "from utils..." works
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.paths import resolve_paths
from utils.seed import set_seed
from utils.io import ensure_dir, save_npy, save_json
from utils.embeddings import load_all_embeddings


# -------------------------
# Data loaders (txt)
# -------------------------
def load_classes(path: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    classes.txt : class_id \\t class_name
    """
    id2label: Dict[int, str] = {}
    label2id: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            cid, name = parts
            cid_int = int(cid)
            id2label[cid_int] = name
            label2id[name] = cid_int
    return id2label, label2id


def load_hierarchy_edges(path: Path) -> List[Tuple[int, int]]:
    """
    class_hierarchy.txt : parent_id \\t child_id
    """
    edges: List[Tuple[int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            p, c = map(int, parts)
            edges.append((p, c))
    return edges


# -------------------------
# Taxonomy helper (노)
# -------------------------
class Taxonomy:
    def __init__(self, num_classes: int, edges: List[Tuple[int, int]]):
        from collections import defaultdict

        self.num_classes = num_classes
        self.children = defaultdict(list)
        self.parent = {cid: None for cid in range(num_classes)}

        for p, c in edges:
            if 0 <= p < num_classes and 0 <= c < num_classes:
                self.children[p].append(c)
                self.parent[c] = p

        self.roots = [cid for cid, p in self.parent.items() if p is None]
        self._path_cache: Dict[int, List[int]] = {}

    def get_ancestors(self, cid: int) -> List[int]:
        res: List[int] = []
        cur = self.parent.get(cid, None)
        while cur is not None:
            res.append(cur)
            cur = self.parent.get(cur, None)
        return res[::-1]

    def get_path(self, cid: int) -> List[int]:
        if cid in self._path_cache:
            return self._path_cache[cid]
        path = self.get_ancestors(cid) + [cid]
        self._path_cache[cid] = path
        return path


# -------------------------
# Graph adjacency (undirected + self loop)
# -------------------------
def build_adj(num_classes: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    adj = np.zeros((num_classes, num_classes), dtype=np.float32)
    for p, c in edges:
        if 0 <= p < num_classes and 0 <= c < num_classes:
            adj[p, c] = 1.0
            adj[c, p] = 1.0
    np.fill_diagonal(adj, 1.0)
    return adj


# -------------------------
# Core class selection (path score)
# -------------------------
def compute_path_scores(
    sim_matrix: np.ndarray,
    taxonomy: Taxonomy,
    num_classes: int,
) -> np.ndarray:
    """
    path_score(c) = mean(sim(doc, node) for node in path(root->...->c))
    sim_matrix: (N, C)
    return: path_scores (N, C)
    """
    N, C = sim_matrix.shape
    assert C == num_classes

    path_scores = np.zeros((N, C), dtype=np.float32)
    for cid in tqdm(range(num_classes), desc="[build_silver] path_scores"):
        path = taxonomy.get_path(cid)
        # sim_matrix[:, path] : (N, len(path))
        path_scores[:, cid] = sim_matrix[:, path].mean(axis=1)
    return path_scores


def select_core_classes(
    path_scores: np.ndarray,
    top_k_core: int = 3,
    min_score: Optional[float] = None,
) -> List[List[int]]:
    """
    문서별 core class: path_score 상위 top_k_core
    """
    N, C = path_scores.shape
    core_per_doc: List[List[int]] = []

    for i in range(N):
        row = path_scores[i]
        idx_sorted = np.argsort(-row)  # desc
        if min_score is not None:
            idx_sorted = idx_sorted[row[idx_sorted] >= min_score]
        core = idx_sorted[:top_k_core].tolist()
        core_per_doc.append(core)

    return core_per_doc


# -------------------------
# Silver labels (core ∪ ancestors(core))
# -------------------------
def build_silver_labels(
    core_classes_per_doc: List[List[int]],
    taxonomy: Taxonomy,
    num_classes: int,
) -> np.ndarray:
    N = len(core_classes_per_doc)
    Y = np.zeros((N, num_classes), dtype=np.float32)

    for i, core_list in enumerate(core_classes_per_doc):
        pos = set()
        for cid in core_list:
            pos.add(int(cid))
            for anc in taxonomy.get_ancestors(int(cid)):
                pos.add(int(anc))
        for cid in pos:
            Y[i, cid] = 1.0
    return Y


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Taxo-style silver labels + adjacency (MPNet embeddings).")

    p.add_argument("--dataset_dir", type=str, required=True,
                   help="~/new/project_release/Amazon_products")

    p.add_argument("--data_dir", type=str, default=None,
                   help="Default: <repo>/data")
    p.add_argument("--outputs_dir", type=str, default=None,
                   help="Default: <repo>/outputs")

    # embeddings
    p.add_argument("--emb_subdir", type=str, default="embeddings_mpnet")
    p.add_argument("--pid_subdir", type=str, default="embeddings_mpnet")
    p.add_argument("--train_doc_name", type=str, default="train_doc_mpnet.npy")
    p.add_argument("--test_doc_name", type=str, default="test_doc_mpnet.npy")
    p.add_argument("--class_name_name", type=str, default="class_name_mpnet.npy")

    # core/silver
    p.add_argument("--top_k_core", type=int, default=3)
    p.add_argument("--min_score", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)

    # performance
    p.add_argument("--batch_size", type=int, default=1024,
                   help="(optional) reserved; not used in this script now (kept for CLI compatibility)")

    # outputs
    p.add_argument("--save_adj_pt", action="store_true",
                   help="Also save adj as .pt (torch tensor) in addition to .npy")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    paths = resolve_paths(
        dataset_dir=args.dataset_dir,
        data_dir=args.data_dir,
        outputs_dir=args.outputs_dir,
        dataset_name="Amazon_products",
    )

    dataset_dir = paths.dataset_dir
    data_dir = paths.data_dir
    outputs_dir = paths.outputs_dir

    print("[build_silver] dataset_dir :", dataset_dir)
    print("[build_silver] data_dir    :", data_dir)
    print("[build_silver] outputs_dir :", outputs_dir)

    # Load class meta
    classes_path = dataset_dir / "classes.txt"
    hierarchy_path = dataset_dir / "class_hierarchy.txt"

    id2label, _ = load_classes(classes_path)
    edges = load_hierarchy_edges(hierarchy_path)
    num_classes = len(id2label)

    taxonomy = Taxonomy(num_classes, edges)
    print("[build_silver] num_classes:", num_classes, "| edges:", len(edges), "| roots:", taxonomy.roots)

    # Load embeddings (MPNet)
    emb = load_all_embeddings(
        dataset_dir=dataset_dir,
        emb_subdir=args.emb_subdir,
        pid_subdir=args.pid_subdir,
        train_doc_name=args.train_doc_name,
        test_doc_name=args.test_doc_name,
        class_name_name=args.class_name_name,
        strict_alignment=False,
        try_load_pid_lists=True,
    )

    train_doc_emb = emb.train_doc_emb.astype(np.float32)
    class_name_emb = emb.class_name_emb.astype(np.float32)

    if train_doc_emb.shape[1] != class_name_emb.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch: train_doc_emb dim={train_doc_emb.shape[1]} vs class_name_emb dim={class_name_emb.shape[1]}"
        )

    print("[build_silver] train_doc_emb:", train_doc_emb.shape, "| class_name_emb:", class_name_emb.shape)
    if emb.pid_list_train is not None:
        print("[build_silver] pid_list_train:", len(emb.pid_list_train))

    # doc-class cosine similarity (normalize_embeddings=True이면 dot==cos)
    sim_matrix = train_doc_emb @ class_name_emb.T  # (N, C)
    sim_matrix = sim_matrix.astype(np.float32)
    print("[build_silver] sim_matrix:", sim_matrix.shape, "| min/max:", float(sim_matrix.min()), float(sim_matrix.max()))

    # path score
    path_scores = compute_path_scores(sim_matrix, taxonomy, num_classes=num_classes)
    print("[build_silver] path_scores:", path_scores.shape)

    # core class per doc
    core_classes_per_doc = select_core_classes(
        path_scores,
        top_k_core=args.top_k_core,
        min_score=args.min_score,
    )

    # silver labels
    Y_silver = build_silver_labels(core_classes_per_doc, taxonomy, num_classes=num_classes)
    print("[build_silver] Y_silver:", Y_silver.shape, "| avg positives/doc:", float(Y_silver.sum(axis=1).mean()))

    # adjacency
    adj = build_adj(num_classes, edges)
    print("[build_silver] adj:", adj.shape, "| nnz:", int(adj.sum()))

    # Save to repo/data (요청대로)
    silver_dir = ensure_dir(data_dir / "silver")
    graph_dir = ensure_dir(data_dir / "graph")

    save_npy(silver_dir / "y_silver.npy", Y_silver)
    save_npy(graph_dir / "adj.npy", adj)

    # core_classes 저장 (pid가 있으면 pid와 같이 저장)
    core_json_path = silver_dir / "core_classes.json"
    core_payload = []
    if emb.pid_list_train is not None:
        for pid, core in zip(emb.pid_list_train.tolist(), core_classes_per_doc):
            core_payload.append({"pid": str(pid), "core": core})
    else:
        for i, core in enumerate(core_classes_per_doc):
            core_payload.append({"idx": int(i), "core": core})

    core_json_path.write_text(json.dumps(core_payload, ensure_ascii=False), encoding="utf-8")

    # optional torch pt
    if args.save_adj_pt:
        adj_pt = torch.from_numpy(adj)
        torch.save(adj_pt, graph_dir / "adj.pt")
        print("[build_silver] saved adj.pt:", graph_dir / "adj.pt")

    meta = {
        "dataset_dir": str(dataset_dir),
        "emb_subdir": args.emb_subdir,
        "pid_subdir": args.pid_subdir,
        "train_doc_name": args.train_doc_name,
        "test_doc_name": args.test_doc_name,
        "class_name_name": args.class_name_name,
        "num_classes": num_classes,
        "edges": len(edges),
        "roots": taxonomy.roots,
        "top_k_core": args.top_k_core,
        "min_score": args.min_score,
        "outputs": {
            "y_silver": str(silver_dir / "y_silver.npy"),
            "core_classes": str(core_json_path),
            "adj_npy": str(graph_dir / "adj.npy"),
            "adj_pt": str(graph_dir / "adj.pt") if args.save_adj_pt else None,
        },
        "shapes": {
            "train_doc_emb": list(train_doc_emb.shape),
            "class_name_emb": list(class_name_emb.shape),
            "sim_matrix": list(sim_matrix.shape),
            "path_scores": list(path_scores.shape),
            "y_silver": list(Y_silver.shape),
            "adj": list(adj.shape),
        },
    }
    save_json(data_dir / "silver_build_meta.json", meta)

    # Quick sanity print
    ex = 0
    ex_core = core_classes_per_doc[ex]
    ex_labels = [id2label[c] for c in ex_core]
    print("[build_silver] example doc 0 core:", ex_core, ex_labels)
    print("[build_silver] saved meta:", data_dir / "silver_build_meta.json")
    print("[build_silver] DONE.")


if __name__ == "__main__":
    main()
