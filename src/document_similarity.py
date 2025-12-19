# src/similarity.py
# -*- coding: utf-8 -*-
"""
Step 1: Document-Class Similarity via Textual Entailment (RoBERTa-MNLI)

PYTHONPATH=.. python test_document_similarity.py \
  --data_dir /home/sagemaker-user/20252R0136DATA30400/project_release/Amazon_products \
  --model roberta-large-mnli \
  --batch_size 8 \
  --max_len 256 \
  --num_docs 3 \
  --num_classes 12 \
  --cache_dir ../outputs/stage1/sim_cache
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _atomic_npz_save(path: str, **arrays) -> None:
    """
    np.savez_compressed adds '.npz' if the filename doesn't end with '.npz'.
    So we ensure temp path ends with '.npz', then atomic replace.
    """
    tmp = f"{path}.tmp.{os.getpid()}.{int(time.time()*1000)}.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _resolve_entailment_index(model) -> int:
    """
    Robustly find entailment label index for MNLI heads.
    Works across common label mappings.
    """
    # 1) Preferred: model.config.label2id (e.g., {'contradiction':0,'neutral':1,'entailment':2})
    label2id = getattr(model.config, "label2id", None)
    if isinstance(label2id, dict) and len(label2id) > 0:
        for k, v in label2id.items():
            if isinstance(k, str) and k.lower() == "entailment":
                return int(v)

    # 2) fallback: id2label (e.g., {0:'CONTRADICTION',1:'NEUTRAL',2:'ENTAILMENT'})
    id2label = getattr(model.config, "id2label", None)
    if isinstance(id2label, dict) and len(id2label) > 0:
        for k, v in id2label.items():
            if isinstance(v, str) and v.lower() == "entailment":
                return int(k)

    # 3) common MNLI convention
    # Many MNLI models use index 2 for entailment
    return 2


@dataclass
class SimBatchResult:
    class_ids: np.ndarray  # (N,) int32
    probs: np.ndarray      # (N,) float32


class DocumentSim:
    """
    Document-to-class similarity using textual entailment.

    Key methods:
      - score_doc(doc_id, doc_text, class_ids, class_names) -> SimBatchResult
      - score_doc_only(doc_text, class_names) -> np.ndarray (entail probs aligned)
      - score_pairs(premises, hypotheses) -> np.ndarray
    """

    def __init__(
        self,
        model_name_or_path: str = "roberta-large-mnli",
        device: Optional[str] = None,
        batch_size: int = 8,
        max_length: int = 256,
        hypothesis_template: str = "This document is about {label}.",
        cache_dir: Optional[str] = None,
        use_amp: bool = True,
        num_workers_tokenize: int = 0,
    ) -> None:
        """
        Args:
          model_name_or_path: HF model for MNLI entailment
          device: "cuda", "cpu", "cuda:0" etc. If None -> auto
          batch_size: inference batch size (you said 8 to avoid OOM)
          max_length: tokenizer max_length
          hypothesis_template: template string for class hypothesis
          cache_dir: if set, enables doc-level disk cache (npz)
          use_amp: mixed precision on CUDA (reduces memory)
          num_workers_tokenize: reserved (not used; kept for future)
        """
        self.model_name_or_path = model_name_or_path
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.hypothesis_template = hypothesis_template
        self.cache_dir = cache_dir
        self.use_amp = bool(use_amp)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if self.cache_dir:
            _ensure_dir(self.cache_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

        self.entail_idx = _resolve_entailment_index(self.model)

    # -------------------------
    # Cache helpers
    # -------------------------

    def _cache_path(self, doc_id: int) -> str:
        assert self.cache_dir is not None
        return os.path.join(self.cache_dir, f"{int(doc_id)}.npz")

    def _load_cache(self, doc_id: int) -> Dict[int, float]:
        """
        Returns {class_id: prob} cached for this document.
        """
        if not self.cache_dir:
            return {}
        path = self._cache_path(doc_id)
        if not os.path.exists(path):
            return {}
        data = np.load(path)
        class_ids = data["class_ids"].astype(np.int32)
        probs = data["probs"].astype(np.float32)
        return {int(c): float(p) for c, p in zip(class_ids.tolist(), probs.tolist())}

    def _save_cache(self, doc_id: int, merged: Dict[int, float]) -> None:
        if not self.cache_dir:
            return
        path = self._cache_path(doc_id)
        # sort by class_id for stable storage
        items = sorted(merged.items(), key=lambda x: x[0])
        class_ids = np.array([k for k, _ in items], dtype=np.int32)
        probs = np.array([v for _, v in items], dtype=np.float32)
        _atomic_npz_save(path, class_ids=class_ids, probs=probs)

    # -------------------------
    # Core scoring API
    # -------------------------

    def make_hypotheses(self, class_names: Sequence[str]) -> List[str]:
        return [self.hypothesis_template.format(label=_normalize_ws(n)) for n in class_names]

    @torch.no_grad()
    def score_pairs(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        """
        Score P(entailment) for each (premise, hypothesis) pair.
        Returns float32 array shape (N,).
        """
        assert len(premises) == len(hypotheses), "premises and hypotheses must have same length"
        n = len(premises)
        if n == 0:
            return np.zeros((0,), dtype=np.float32)

        out_probs: List[np.ndarray] = []

        # Choose autocast dtype only when CUDA
        use_cuda = self.device.startswith("cuda")
        autocast_enabled = self.use_amp and use_cuda
        autocast_dtype = torch.float16  # safe default for inference

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            p_batch = [_normalize_ws(x) for x in premises[start:end]]
            h_batch = [_normalize_ws(x) for x in hypotheses[start:end]]

            enc = self.tokenizer(
                p_batch,
                h_batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,  # pad within batch
                return_tensors="pt",
            )

            enc = {k: v.to(self.device) for k, v in enc.items()}

            if autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    logits = self.model(**enc).logits
            else:
                logits = self.model(**enc).logits

            probs = torch.softmax(logits, dim=-1)[:, self.entail_idx]
            out_probs.append(probs.detach().float().cpu().numpy())

        return np.concatenate(out_probs, axis=0).astype(np.float32)

    def score_doc_only(self, doc_text: str, class_names: Sequence[str]) -> np.ndarray:
        """
        Convenience: score a single doc against a list of class_names.
        Returns entailment probs aligned with class_names.
        """
        hypotheses = self.make_hypotheses(class_names)
        premises = [doc_text] * len(class_names)
        return self.score_pairs(premises, hypotheses)

    def score_doc(
        self,
        doc_id: int,
        doc_text: str,
        class_ids: Sequence[int],
        class_names: Sequence[str],
        update_cache: bool = True,
    ) -> SimBatchResult:
        """
        Score a single document against selected classes, with optional caching.

        Args:
          doc_id: document id (for cache file naming)
          doc_text: raw document text
          class_ids: class ids requested (aligned with class_names)
          class_names: class names requested (aligned with class_ids)
          update_cache: if True, save merged cache to disk

        Returns:
          SimBatchResult with arrays aligned with input class_ids order.
        """
        assert len(class_ids) == len(class_names), "class_ids and class_names must align"
        class_ids_int = [int(x) for x in class_ids]

        # no cache mode
        if not self.cache_dir:
            probs = self.score_doc_only(doc_text, class_names)
            return SimBatchResult(class_ids=np.array(class_ids_int, dtype=np.int32), probs=probs)

        cached = self._load_cache(doc_id)
        missing_ids: List[int] = []
        missing_names: List[str] = []
        for cid, cname in zip(class_ids_int, class_names):
            if cid not in cached:
                missing_ids.append(cid)
                missing_names.append(cname)

        if missing_ids:
            new_probs = self.score_doc_only(doc_text, missing_names)
            for cid, p in zip(missing_ids, new_probs.tolist()):
                cached[int(cid)] = float(p)

            if update_cache:
                self._save_cache(doc_id, cached)

        probs_out = np.array([cached[int(cid)] for cid in class_ids_int], dtype=np.float32)
        return SimBatchResult(class_ids=np.array(class_ids_int, dtype=np.int32), probs=probs_out)

    # -------------------------
    # Bulk scoring utilities (optional)
    # -------------------------

    def score_docs_against_classes(
        self,
        doc_ids: Sequence[int],
        doc_texts: Sequence[str],
        class_ids: Sequence[int],
        class_names: Sequence[str],
        update_cache: bool = True,
    ) -> Dict[int, SimBatchResult]:
        """
        Score multiple docs against the same class set.
        Uses per-doc caching; does not attempt to batch across docs/classes globally
        (keeps memory predictable).
        """
        assert len(doc_ids) == len(doc_texts)
        results: Dict[int, SimBatchResult] = {}
        for did, txt in zip(doc_ids, doc_texts):
            results[int(did)] = self.score_doc(
                doc_id=int(did),
                doc_text=txt,
                class_ids=class_ids,
                class_names=class_names,
                update_cache=update_cache,
            )
        return results


# -------------------------
# Optional quick test
# -------------------------
if __name__ == "__main__":
    # Minimal smoke test without needing full pipeline
    sim = DocumentSim(
        model_name_or_path="roberta-large-mnli",
        batch_size=8,
        max_length=256,
        cache_dir=None,
    )
    doc = "This is a document about beef jerky snacks and dried meat products."
    class_names = ["meat_poultry", "jerky", "toothbrush"]
    probs = sim.score_doc_only(doc, class_names)
    for c, p in zip(class_names, probs.tolist()):
        print(f"{c:20s} entail_prob={p:.4f}")
