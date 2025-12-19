# base/data/dataloader.py
# -*- coding: utf-8 -*-
"""
TaxoClass dataloader (enhanced version, renamed corpus APIs)

Changes requested
-----------------
- Renamed DocumentCorpus -> DocStore
- Renamed "get_*" methods to cleaner names:
    DocStore.by_id(doc_id)
    DocStore.text(doc_id)
    DocStore.texts()
    DocStore.doc_ids()
- TaxoDataset method get_by_doc_id -> by_id

Everything else remains compatible with the previous enhanced design.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# -----------------------------
# 1) Data containers
# -----------------------------

@dataclass(frozen=True)
class Document:
    doc_id: int
    text: str


class DocStore:
    """
    Convenience wrapper around a list of Documents.

    Key feature:
      - doc_id -> index lookup is safe (does NOT assume doc_id == index).

    API (no "get" prefix):
      - by_id(doc_id)   -> Document | None
      - text(doc_id)    -> str | None
      - texts()         -> List[str]
      - doc_ids()       -> List[int]
    """

    def __init__(self, docs: Sequence[Document]):
        self.docs: List[Document] = list(docs)
        self._id2idx: Dict[int, int] = {d.doc_id: i for i, d in enumerate(self.docs)}

    def __len__(self) -> int:
        return len(self.docs)

    def __getitem__(self, idx: int) -> Document:
        return self.docs[idx]

    def by_id(self, doc_id: int) -> Optional[Document]:
        i = self._id2idx.get(int(doc_id))
        return None if i is None else self.docs[i]

    def text(self, doc_id: int) -> Optional[str]:
        d = self.by_id(doc_id)
        return None if d is None else d.text

    def texts(self) -> List[str]:
        return [d.text for d in self.docs]

    def doc_ids(self) -> List[int]:
        return [d.doc_id for d in self.docs]


# -----------------------------
# 2) Taxonomy helper (DAG)
# -----------------------------

class Taxonomy:
    """
    Minimal DAG taxonomy utility:
    - class_id <-> class_name mapping
    - parent/child adjacency
    - ancestor / descendant closure (memoized)
    - sibling set
    """

    def __init__(
        self,
        id2name: Dict[int, str],
        parents: Dict[int, Set[int]],
        children: Dict[int, Set[int]],
    ) -> None:
        self.id2name = dict(id2name)
        self.name2id = {v: k for k, v in self.id2name.items()}
        self.parents = {k: set(v) for k, v in parents.items()}
        self.children = {k: set(v) for k, v in children.items()}

        self.num_classes = len(self.id2name)
        self._anc_cache: Dict[int, Set[int]] = {}
        self._desc_cache: Dict[int, Set[int]] = {}
        self._sib_cache: Dict[int, Set[int]] = {}

        for cid in range(self.num_classes):
            self.parents.setdefault(cid, set())
            self.children.setdefault(cid, set())

    @staticmethod
    def load_from_dir(data_dir: str) -> "Taxonomy":
        classes_path = os.path.join(data_dir, "classes.txt")
        hier_path = os.path.join(data_dir, "class_hierarchy.txt")
        return Taxonomy.load(classes_path, hier_path)

    @staticmethod
    def load(classes_path: str, hierarchy_path: str) -> "Taxonomy":
        id2name: Dict[int, str] = {}
        with open(classes_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cid, cname = _split_id_text(line)
                id2name[int(cid)] = cname

        parents: Dict[int, Set[int]] = {cid: set() for cid in id2name.keys()}
        children: Dict[int, Set[int]] = {cid: set() for cid in id2name.keys()}

        with open(hierarchy_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p, c = _split_two_ints(line)
                parents.setdefault(c, set()).add(p)
                children.setdefault(p, set()).add(c)

        num_classes = len(id2name)
        for cid in range(num_classes):
            parents.setdefault(cid, set())
            children.setdefault(cid, set())

        return Taxonomy(id2name=id2name, parents=parents, children=children)

    def ancestors(self, c: int) -> Set[int]:
        c = int(c)
        if c in self._anc_cache:
            return self._anc_cache[c]
        out: Set[int] = set()
        stack = list(self.parents.get(c, set()))
        while stack:
            p = stack.pop()
            if p in out:
                continue
            out.add(p)
            stack.extend(self.parents.get(p, set()))
        self._anc_cache[c] = out
        return out

    def descendants(self, c: int) -> Set[int]:
        c = int(c)
        if c in self._desc_cache:
            return self._desc_cache[c]
        out: Set[int] = set()
        stack = list(self.children.get(c, set()))
        while stack:
            ch = stack.pop()
            if ch in out:
                continue
            out.add(ch)
            stack.extend(self.children.get(ch, set()))
        self._desc_cache[c] = out
        return out

    def siblings(self, c: int) -> Set[int]:
        c = int(c)
        if c in self._sib_cache:
            return self._sib_cache[c]
        sibs: Set[int] = set()
        for p in self.parents.get(c, set()):
            for ch in self.children.get(p, set()):
                if ch != c:
                    sibs.add(ch)
        self._sib_cache[c] = sibs
        return sibs

    def all_class_ids(self) -> List[int]:
        return list(range(self.num_classes))


# -----------------------------
# 3) Label builders (vector)
# -----------------------------

def build_train_label_vector(core_ids: Sequence[int], taxonomy: Taxonomy) -> np.ndarray:
    """
    Hierarchy-aware training labels in {1,0,-1}:
      1 = core ∪ ancestors(core)
      0 = all - pos - descendants(core)
     -1 = ignore (typically descendants(core) or unknown)
    """
    C = taxonomy.num_classes
    y = np.full((C,), -1, dtype=np.int8)

    core_set = {int(x) for x in core_ids if x is not None}
    if not core_set:
        return y

    pos: Set[int] = set()
    desc: Set[int] = set()

    for c in core_set:
        if 0 <= c < C:
            pos.add(c)
            pos.update(taxonomy.ancestors(c))
            desc.update(taxonomy.descendants(c))

    all_ids = set(taxonomy.all_class_ids())
    neg = all_ids.difference(pos).difference(desc)

    for c in neg:
        y[c] = 0
    for c in pos:
        y[c] = 1

    return y


def build_eval_label_vector(gt_ids: Sequence[int], taxonomy: Taxonomy) -> np.ndarray:
    """
    Eval/analysis labels in {0,1}:
      1 for GT labels and all ancestors, else 0.
    """
    C = taxonomy.num_classes
    y = np.zeros((C,), dtype=np.int8)

    gt_set = {int(x) for x in gt_ids if x is not None}
    for c in gt_set:
        if 0 <= c < C:
            y[c] = 1
            for a in taxonomy.ancestors(c):
                if 0 <= a < C:
                    y[a] = 1
    return y


# -----------------------------
# 4) Label builders (matrix)
# -----------------------------

def build_train_label_matrix(
    docs: Sequence[Document],
    core_map: Dict[int, List[int]],
    taxonomy: Taxonomy,
    drop_empty_core: bool = False,
) -> Tuple[np.ndarray, List[int]]:
    rows: List[np.ndarray] = []
    kept_doc_ids: List[int] = []

    for d in docs:
        core = core_map.get(d.doc_id, [])
        if drop_empty_core and (not core):
            continue
        rows.append(build_train_label_vector(core, taxonomy))
        kept_doc_ids.append(d.doc_id)

    if not rows:
        return np.zeros((0, taxonomy.num_classes), dtype=np.int8), []
    return np.stack(rows, axis=0), kept_doc_ids


def build_eval_label_matrix(
    docs: Sequence[Document],
    gt_map: Dict[int, List[int]],
    taxonomy: Taxonomy,
) -> Tuple[np.ndarray, List[int]]:
    rows: List[np.ndarray] = []
    doc_ids: List[int] = []

    for d in docs:
        gt = gt_map.get(d.doc_id, [])
        rows.append(build_eval_label_vector(gt, taxonomy))
        doc_ids.append(d.doc_id)

    if not rows:
        return np.zeros((0, taxonomy.num_classes), dtype=np.int8), []
    return np.stack(rows, axis=0), doc_ids


# -----------------------------
# 5) Corpus + label readers
# -----------------------------

def read_corpus(corpus_path: str) -> List[Document]:
    docs: List[Document] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            doc_id_str, text = _split_id_text(line)
            docs.append(Document(doc_id=int(doc_id_str), text=text))
    return docs


def read_core_labels_jsonl(core_jsonl_path: str) -> Dict[int, List[int]]:
    """
    JSONL row variants supported:
      {"doc_id": 123, "core_ids": [1,9,33]}
      {"doc_id": 123, "core": [...]}
      {"id": 123, "core_ids": [...]}
      {"id": 123, "core": [...]}
    """
    out: Dict[int, List[int]] = {}
    with open(core_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            did = obj.get("doc_id", obj.get("id"))
            core = obj.get("core_ids", obj.get("core"))
            if did is None or core is None:
                continue
            out[int(did)] = [int(x) for x in core]
    return out


def read_gt_labels(gt_path: str) -> Dict[int, List[int]]:
    """
    Flexible GT label reader (dev/analysis):
      - TSV: doc_id<TAB>label_ids (comma/space separated)
      - JSONL: {"doc_id":12,"labels":[...]} or {"id":12,"labels":[...]}
    """
    if gt_path.lower().endswith(".jsonl"):
        out: Dict[int, List[int]] = {}
        with open(gt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                did = obj.get("doc_id", obj.get("id"))
                labels = obj.get("labels")
                if did is None or labels is None:
                    continue
                out[int(did)] = [int(x) for x in labels]
        return out

    out2: Dict[int, List[int]] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            did_str, lab_str = _split_id_text(line)
            out2[int(did_str)] = _parse_int_list(lab_str)
    return out2


# -----------------------------
# 6) Keywords loader (optional)
# -----------------------------

def load_class_keywords(keywords_file: str) -> Dict[str, List[str]]:
    """
    Expected line format:
      class_name: kw1, kw2, kw3
    """
    keywords_dict: Dict[str, List[str]] = {}
    with open(keywords_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            class_name = parts[0].strip()
            kws = [kw.strip() for kw in parts[1].split(",") if kw.strip()]
            keywords_dict[class_name] = kws
    return keywords_dict


# -----------------------------
# 7) Dataset
# -----------------------------

class TaxoDataset(Dataset):
    """
    Supplies tokenized inputs and labels.

    Modes:
      - "train": labels in {1,0,-1} from core_labels
      - "eval" : labels in {0,1}   from gt_labels (incl. ancestors)
      - "infer": no labels
    """

    def __init__(
        self,
        docs: Sequence[Document],
        taxonomy: Taxonomy,
        tokenizer_name_or_path: str,
        max_length: int = 256,
        mode: str = "train",
        core_labels: Optional[Dict[int, List[int]]] = None,
        gt_labels: Optional[Dict[int, List[int]]] = None,
        drop_empty_core: bool = True,
        pad_to_max_length: bool = False,
    ) -> None:
        assert mode in {"train", "eval", "infer"}, f"invalid mode: {mode}"
        self.taxonomy = taxonomy
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
        self.max_length = int(max_length)
        self.mode = mode
        self.pad_to_max_length = bool(pad_to_max_length)

        self.core_labels = core_labels or {}
        self.gt_labels = gt_labels or {}

        self.docs = list(docs)
        self._id2idx: Dict[int, int] = {d.doc_id: i for i, d in enumerate(self.docs)}

        if self.mode == "train" and drop_empty_core:
            filtered = [d for d in self.docs if self.core_labels.get(d.doc_id)]
            self.docs = filtered
            self._id2idx = {d.doc_id: i for i, d in enumerate(self.docs)}

    def __len__(self) -> int:
        return len(self.docs)

    def by_id(self, doc_id: int) -> Optional[Document]:
        i = self._id2idx.get(int(doc_id))
        return None if i is None else self.docs[i]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        d = self.docs[idx]
        enc = self.tokenizer(
            d.text,
            max_length=self.max_length,
            truncation=True,
            padding=("max_length" if self.pad_to_max_length else False),
            return_tensors=None,
        )

        item: Dict[str, Any] = {
            "doc_id": d.doc_id,
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        }

        if self.mode == "train":
            core = self.core_labels.get(d.doc_id, [])
            item["labels"] = torch.tensor(build_train_label_vector(core, self.taxonomy), dtype=torch.int8)

        elif self.mode == "eval":
            gt = self.gt_labels.get(d.doc_id, [])
            item["labels"] = torch.tensor(build_eval_label_vector(gt, self.taxonomy), dtype=torch.int8)

        return item

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        doc_ids = torch.tensor([x["doc_id"] for x in batch], dtype=torch.long)

        input_ids = [x["input_ids"] for x in batch]
        attn = [x["attention_mask"] for x in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        attn = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)

        out: Dict[str, Any] = {
            "doc_id": doc_ids,
            "input_ids": input_ids,
            "attention_mask": attn,
        }

        if "labels" in batch[0]:
            labels = torch.stack([x["labels"] for x in batch], dim=0)  # int8 [B,C]
            out["labels"] = labels

            # Always provide mask/float for simpler training code
            label_mask = (labels != -1).float()
            labels_float = labels.clone().float()
            labels_float[labels_float < 0] = 0.0

            out["labels_float"] = labels_float
            out["label_mask"] = label_mask

        return out


# -----------------------------
# 8) Parsing helpers
# -----------------------------

_WS_SPLIT = re.compile(r"\s+")


def _split_id_text(line: str) -> Tuple[str, str]:
    if "\t" in line:
        a, b = line.split("\t", 1)
        return a.strip(), b.strip()
    parts = _WS_SPLIT.split(line.strip(), maxsplit=1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def _split_two_ints(line: str) -> Tuple[int, int]:
    if "\t" in line:
        a, b = line.split("\t", 1)
    else:
        parts = _WS_SPLIT.split(line.strip(), maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Cannot parse two ints from line: {line!r}")
        a, b = parts
    return int(a.strip()), int(b.strip())


def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    toks = re.split(r"[,\s]+", s)
    out: List[int] = []
    for t in toks:
        if t:
            out.append(int(t))
    return out
