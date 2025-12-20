# src/make_embeddings_mpnet.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Ensure "src/" is on sys.path so "from utils..." works when running:
#   python3 src/make_embeddings_mpnet.py ...
SRC_DIR = Path(__file__).resolve().parent  # .../src
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.paths import resolve_paths
from utils.seed import set_seed
from utils.io import ensure_dir, save_npy, save_json


def load_corpus(path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    corpus file: pid \\t text
    returns:
      pid_list: np.ndarray[str] shape (N,)
      texts: list[str] length N in pid_list order
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing corpus file: {path}")

    pids: List[str] = []
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            pid, text = parts
            pids.append(pid)
            texts.append(text)
    return np.array(pids, dtype=object), texts


def load_classes(path: Path) -> Dict[int, str]:
    """
    classes.txt: class_id \\t class_name
    return: {class_id: class_name}
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing classes file: {path}")

    id2label: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            cid = int(parts[0])
            name = parts[1]
            id2label[cid] = name
    return id2label


def load_keywords(path: Path, label2id: Dict[str, int]) -> Dict[int, List[str]]:
    """
    class_related_keywords.txt : CLASS_NAME: kw1, kw2,...
    return: {class_id: [kw...]}
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing keywords file: {path}")

    d: Dict[int, List[str]] = {cid: [] for cid in label2id.values()}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            name, kws = line.strip().split(":", 1)
            kws_list = [k.strip() for k in kws.split(",") if k.strip()]
            if name in label2id:
                cid = label2id[name]
                d[cid] = kws_list
    return d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create MPNet embeddings (all-mpnet-base-v2) and save as .npy under dataset_dir/embeddings_mpnet."
    )
    p.add_argument("--dataset_dir", type=str, default=None,
                   help="Dataset root dir (must contain train/, test/, classes.txt, class_related_keywords.txt). "
                        "Example: ~/new/project_release/Amazon_products")

    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--outputs_dir", type=str, default=None)

    p.add_argument("--model_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_seq_length", type=int, default=128)

    p.add_argument("--emb_subdir", type=str, default="embeddings_mpnet",
                   help="Output subdir under dataset_dir (default: embeddings_mpnet)")

    p.add_argument("--seed", type=int, default=42)

    # caching / overwrite behavior
    p.add_argument("--force", action="store_true",
                   help="If set, recompute and overwrite existing npy files. "
                        "Default: skip if all expected files already exist.")
    return p.parse_args()


def expected_files() -> List[str]:
    """
    Files produced by this script (mirrors your notebook naming).
    """
    return [
        "train_doc_mpnet.npy",
        "test_doc_mpnet.npy",
        "class_name_mpnet.npy",
        "class_kw_mpnet.npy",
        "pid_list_train.npy",
        "pid_list_test.npy",
        "emb_meta.json",
    ]


def all_exist(emb_dir: Path) -> Tuple[bool, List[Path]]:
    missing: List[Path] = []
    for fn in expected_files():
        p = emb_dir / fn
        if not p.exists():
            missing.append(p)
    return (len(missing) == 0), missing


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Resolve canonical paths
    paths = resolve_paths(
        dataset_dir=args.dataset_dir,
        data_dir=args.data_dir,
        outputs_dir=args.outputs_dir,
        dataset_name="Amazon_products",
    )
    dataset_dir = paths.dataset_dir

    # Input paths
    train_corpus_path = dataset_dir / "train" / "train_corpus.txt"
    test_corpus_path = dataset_dir / "test" / "test_corpus.txt"
    classes_path = dataset_dir / "classes.txt"
    keywords_path = dataset_dir / "class_related_keywords.txt"

    # Output dir
    emb_dir = ensure_dir(dataset_dir / args.emb_subdir)

    # Cache behavior
    ok, missing = all_exist(emb_dir)
    if ok and not args.force:
        print("[make_embeddings_mpnet] All embedding files already exist. Skip.")
        print("[make_embeddings_mpnet] To recompute, pass --force.")
        for fn in expected_files():
            print("  -", emb_dir / fn)
        return

    if not ok:
        print("[make_embeddings_mpnet] Missing files detected; will compute:")
        for p in missing:
            print("  -", p)

    print("[make_embeddings_mpnet] dataset_dir:", dataset_dir)
    print("[make_embeddings_mpnet] emb_dir    :", emb_dir)
    print("[make_embeddings_mpnet] model_name :", args.model_name)
    print("[make_embeddings_mpnet] batch_size :", args.batch_size)
    print("[make_embeddings_mpnet] max_seq_length:", args.max_seq_length)

    # Load raw data
    pid_train, train_texts = load_corpus(train_corpus_path)
    pid_test, test_texts = load_corpus(test_corpus_path)

    id2label = load_classes(classes_path)
    num_classes = max(id2label.keys()) + 1

    label2id = {name: cid for cid, name in id2label.items()}
    label_keywords = load_keywords(keywords_path, label2id)

    class_ids = list(range(num_classes))
    class_names = [id2label[cid].replace("_", " ") for cid in class_ids]

    merged_class_texts: List[str] = []
    for cid in class_ids:
        name = id2label[cid].replace("_", " ")
        kws = label_keywords.get(cid, [])
        if kws:
            merged = name + " : " + ", ".join(kws)
        else:
            merged = name
        merged_class_texts.append(merged)

    print(f"[make_embeddings_mpnet] train/test/classes: {len(train_texts)} / {len(test_texts)} / {num_classes}")

    # sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers is required. Install with: pip install -U sentence-transformers"
        ) from e

    model = SentenceTransformer(args.model_name)
    model.max_seq_length = args.max_seq_length

    # Encode (normalize_embeddings=True matches your notebook; dot == cosine)
    train_doc_mpnet = model.encode(
        train_texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    test_doc_mpnet = model.encode(
        test_texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    class_name_mpnet = model.encode(
        class_names,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    class_kw_mpnet = model.encode(
        merged_class_texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # Save npy (overwrite if exists; user controlled by --force/skip)
    save_npy(emb_dir / "train_doc_mpnet.npy", train_doc_mpnet)
    save_npy(emb_dir / "test_doc_mpnet.npy", test_doc_mpnet)
    save_npy(emb_dir / "class_name_mpnet.npy", class_name_mpnet)
    save_npy(emb_dir / "class_kw_mpnet.npy", class_kw_mpnet)
    save_npy(emb_dir / "pid_list_train.npy", pid_train)
    save_npy(emb_dir / "pid_list_test.npy", pid_test)

    meta = {
        "dataset_dir": str(dataset_dir),
        "emb_dir": str(emb_dir),
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "normalize_embeddings": True,
        "shapes": {
            "train_doc_mpnet": list(train_doc_mpnet.shape),
            "test_doc_mpnet": list(test_doc_mpnet.shape),
            "class_name_mpnet": list(class_name_mpnet.shape),
            "class_kw_mpnet": list(class_kw_mpnet.shape),
            "pid_train": int(len(pid_train)),
            "pid_test": int(len(pid_test)),
        },
    }
    save_json(emb_dir / "emb_meta.json", meta)

    print("[make_embeddings_mpnet] saved:")
    for fn in expected_files():
        print("  -", emb_dir / fn)


if __name__ == "__main__":
    main()
