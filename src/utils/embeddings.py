# src/utils/embeddings.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np


@dataclass(frozen=True)
class EmbeddingBundle:
    train_doc_emb: np.ndarray
    test_doc_emb: np.ndarray
    class_name_emb: np.ndarray
    pid_list_train: Optional[np.ndarray] = None
    pid_list_test: Optional[np.ndarray] = None
    # helpful meta
    train_path: Optional[str] = None
    test_path: Optional[str] = None
    class_path: Optional[str] = None


def _load_npy(path: Path, allow_pickle: bool = False) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path, allow_pickle=allow_pickle)


def _glob_pick_one(emb_dir: Path, patterns: List[str], kind: str) -> Path:
    """
    Pick one file by glob. If multiple matches, raise with list for debugging.
    """
    matches: List[Path] = []
    for pat in patterns:
        matches.extend(sorted(emb_dir.glob(pat)))

    # light filtering to avoid accidental picks
    if kind in ("train", "test"):
        matches = [m for m in matches if ("class" not in m.name and "label" not in m.name)]
    if kind == "class":
        matches = [m for m in matches if ("class" in m.name or "label" in m.name)]

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not infer {kind} embedding in {emb_dir}. "
            f"Tried patterns: {patterns}"
        )

    # ambiguous
    lines = [f"Ambiguous {kind} embedding candidates in {emb_dir}:"]
    lines += [f"  - {m.name}" for m in matches[:80]]
    if len(matches) > 80:
        lines.append(f"  ... (+{len(matches)-80} more)")
    lines.append("Fix: pass explicit file name via --train_doc_name/--test_doc_name/--class_name_name.")
    raise FileNotFoundError("\n".join(lines))


def _resolve_embedding_file(emb_dir: Path, preferred_name: Optional[str], kind: str) -> Path:
    """
    Resolve one embedding file path.
    - if preferred_name exists -> use it
    - else infer via globs
    """
    if preferred_name:
        p = emb_dir / preferred_name
        if p.exists():
            return p

    # fallback inference patterns
    if kind == "train":
        patterns = ["train_doc*.npy", "train*doc*.npy"]
    elif kind == "test":
        patterns = ["test_doc*.npy", "test*doc*.npy"]
    elif kind == "class":
        patterns = ["class_name*.npy", "class*name*.npy", "class_emb*.npy", "label*.npy", "class*.npy"]
    else:
        patterns = ["*.npy"]

    return _glob_pick_one(emb_dir, patterns, kind=kind)


def embeddings_exist(
    dataset_dir: str | Path,
    emb_subdir: str,
    train_doc_name: str,
    test_doc_name: str,
    class_name_name: str,
) -> Tuple[bool, List[str]]:
    """
    Returns (ok, missing_paths_relative_to_dataset_dir)
    """
    dataset_dir = Path(dataset_dir)
    emb_dir = dataset_dir / emb_subdir
    missing: List[str] = []

    for fn in [train_doc_name, test_doc_name, class_name_name]:
        p = emb_dir / fn
        if not p.exists():
            missing.append(str(p.relative_to(dataset_dir)))

    return (len(missing) == 0), missing


def load_embeddings(
    dataset_dir: str | Path,
    emb_subdir: str = "embeddings_mpnet",
    train_doc_name: Optional[str] = None,
    test_doc_name: Optional[str] = None,
    class_name_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
    """
    Load (train_doc_emb, test_doc_emb, class_name_emb) from dataset_dir/emb_subdir.

    If names are None or missing, try to infer with glob patterns.
    Returns arrays + resolved absolute path strings.
    """
    dataset_dir = Path(dataset_dir)
    emb_dir = dataset_dir / emb_subdir
    if not emb_dir.exists():
        raise FileNotFoundError(f"Embedding directory does not exist: {emb_dir}")

    train_path = _resolve_embedding_file(emb_dir, train_doc_name, kind="train")
    test_path = _resolve_embedding_file(emb_dir, test_doc_name, kind="test")
    class_path = _resolve_embedding_file(emb_dir, class_name_name, kind="class")

    train_doc_emb = _load_npy(train_path)
    test_doc_emb = _load_npy(test_path)
    class_name_emb = _load_npy(class_path)

    return train_doc_emb, test_doc_emb, class_name_emb, str(train_path), str(test_path), str(class_path)


def load_pid_lists(
    dataset_dir: str | Path,
    pid_subdir: str = "embeddings_mpnet",
    pid_train_name: str = "pid_list_train.npy",
    pid_test_name: str = "pid_list_test.npy",
) -> Tuple[np.ndarray, np.ndarray]:
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
    errs: List[str] = []

    if pid_list_train is not None and len(pid_list_train) != train_doc_emb.shape[0]:
        errs.append(
            f"pid_list_train length ({len(pid_list_train)}) != train_doc_emb rows ({train_doc_emb.shape[0]})"
        )
    if pid_list_test is not None and len(pid_list_test) != test_doc_emb.shape[0]:
        errs.append(
            f"pid_list_test length ({len(pid_list_test)}) != test_doc_emb rows ({test_doc_emb.shape[0]})"
        )

    if errs:
        msg = "Embedding/pid alignment check failed:\n- " + "\n- ".join(errs)
        if strict:
            raise ValueError(msg)
        print(msg)


def load_all_embeddings(
    dataset_dir: str | Path,
    emb_subdir: str = "embeddings_mpnet",
    pid_subdir: str = "embeddings_mpnet",
    train_doc_name: Optional[str] = None,
    test_doc_name: Optional[str] = None,
    class_name_name: Optional[str] = None,
    strict_alignment: bool = False,
    try_load_pid_lists: bool = True,
) -> EmbeddingBundle:
    """
    One-stop loader used by build_silver/train/predict.

    - emb_subdir points to embeddings folder (mpnet/gte/etc)
    - pid_subdir points to pid lists folder (often same as emb_subdir for mpnet)
    - file names are optional; if omitted/missing, fallback inference is used
    """
    train_doc_emb, test_doc_emb, class_name_emb, train_p, test_p, class_p = load_embeddings(
        dataset_dir=dataset_dir,
        emb_subdir=emb_subdir,
        train_doc_name=train_doc_name,
        test_doc_name=test_doc_name,
        class_name_name=class_name_name,
    )

    pid_train = None
    pid_test = None
    if try_load_pid_lists:
        try:
            pid_train, pid_test = load_pid_lists(dataset_dir=dataset_dir, pid_subdir=pid_subdir)
        except FileNotFoundError:
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
        train_path=train_p,
        test_path=test_p,
        class_path=class_p,
    )
