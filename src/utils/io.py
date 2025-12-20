# src/utils/io.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    """
    Append one JSON object per line.
    Useful for logging / experiment traces.
    """
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_npy(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, arr)


def load_npy(path: str | Path, allow_pickle: bool = False) -> np.ndarray:
    return np.load(Path(path), allow_pickle=allow_pickle)


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(Path(path))
    return {k: data[k] for k in data.files}


def try_import_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None


def torch_save(path: str | Path, obj: Any) -> None:
    torch = try_import_torch()
    if torch is None:
        raise RuntimeError("torch is not available, cannot torch.save()")
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(obj, path)


def torch_load(path: str | Path, map_location: Optional[str] = None) -> Any:
    torch = try_import_torch()
    if torch is None:
        raise RuntimeError("torch is not available, cannot torch.load()")
    return torch.load(Path(path), map_location=map_location)
