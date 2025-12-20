# src/utils/embeddings.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class EmbeddingBundle:
    """
    Container for embeddings + pid lists (when available).
    """
    train_doc_emb: np.ndarray  # (N_train, dim)
    test_doc_emb: np.ndarray   # (N_test, dim)
    class_name_emb: np.ndarray # (C, dim)
    pid_list_train: Optional[np.ndarray] = None
    pid_list_test: Optional[np.ndarray] = None


def _load_npy(path: Path, allow_pickle: bool = False) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path, allow_pickle=allow_pickle)


def load_embeddings(
    dataset_dir: str | Path,
    emb_subdir: str = "embeddings",
    train_doc_name: str = "train_doc_gte.npy",
    test_doc_name: str = "test_doc_gte.npy",
    class_name_name: str = "class_name_gte.npy",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load GTE embeddings from:
      {dataset_dir}/{emb_subdir}/train_doc_gte.npy
      {dataset_dir}/{emb_subdir}/test_doc_gte.npy
      {dataset_dir}/{emb_subdir}/class_name_gte.npy
    """
    dataset_dir = Path(dataset_dir)
    emb_dir = dataset_dir / emb_subdir

    train_doc_emb = _load_npy(emb_dir / train_doc_name)
    test_doc_emb = _load_npy(emb_dir / test_doc_name)
    class_name_emb = _load_npy(emb_dir / class_name_name)

    return train_doc_emb, test_doc_emb, class_name_emb


def load_pid_lists(
    dataset_dir: str | Path,
    pid_subdir: str = "embeddings_mpnet",
    pid_train_name: str = "pid_list_train.npy",
    pid_test_name: str = "pid_list_test.npy",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pid lists. In your notebook they were stored under embeddings_mpnet/.
    """
    dataset_dir = Path(dataset_dir)
    pid_dir = dataset_dir / pid_subdir

    pid_train = _load_npy(pid_dir / pid_train_name, allow_pickle=True)
    pid_test = _load_npy(pid_dir / pid_test_name, allow_pickle=True)
    return pid_train, pid_test


def validate_alignment(
    train_doc_emb: np.ndarray,
    test_doc_emb: np.ndarray,
    pid_list_train: Optional[np.ndarray],
    pid_list_test: Optional[np.ndarray],
    strict: bool = True,
) -> None:
    """
    Sanity-check that pid lists align with embeddings sizes.
    This is important because your pipeline loads pid_list from a different subdir.
    """
    errors = []

    if pid_list_train is not None and len(pid_list_train) != train_doc_emb.shape[0]:
        errors.append(
            f"pid_list_train length ({len(pid_list_train)}) != train_doc_emb rows ({train_doc_emb.shape[0]})"
        )
    if pid_list_test is not None and len(pid_list_test) != test_doc_emb.shape[0]:
        errors.append(
            f"pid_list_test length ({len(pid_list_test)}) != test_doc_emb rows ({test_doc_emb.shape[0]})"
        )

    if errors:
        msg = "Embedding/pid alignment check failed:\n- " + "\n- ".join(errors)
        if strict:
            raise ValueError(msg)
        else:
            print(msg)


def load_all_embeddings(
    dataset_dir: str | Path,
    emb_subdir: str = "embeddings",
    pid_subdir: str = "embeddings_mpnet",
    strict_alignment: bool = True,
) -> EmbeddingBundle:
    """
    High-level loader:
      - loads gte embeddings from emb_subdir
      - loads pid lists from pid_subdir (if available)
      - validates sizes

    If pid list files are missing, it will set them to None.
    """
    train_doc_emb, test_doc_emb, class_name_emb = load_gte_embeddings(
        dataset_dir=dataset_dir,
        emb_subdir=emb_subdir,
    )

    pid_train = None
    pid_test = None
    try:
        pid_train, pid_test = load_pid_lists(dataset_dir=dataset_dir, pid_subdir=pid_subdir)
    except FileNotFoundError:
        # pid lists are optional; pipeline can still run without them
        pid_train, pid_test = None, None

    validate_alignment(
        train_doc_emb=train_doc_emb,
        test_doc_emb=test_doc_emb,
        pid_list_train=pid_train,
        pid_list_test=pid_test,
        strict=strict_alignment,
    )

    return EmbeddingBundle(
        train_doc_emb=train_doc_emb,
        test_doc_emb=test_doc_emb,
        class_name_emb=class_name_emb,
        pid_list_train=pid_train,
        pid_list_test=pid_test,
    )
