# data/test_dataloader.py
# 데이터로드 잘 되는지 테스트
"""
python data/test_dataloader.py \
  --data_dir /home/sagemaker-user/20252R0136DATA30400/project_release/Amazon_products \
  --tok bert-base-uncased \
  --max_len 128 \
  --batch_size 4"""

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import (
    Taxonomy,
    read_corpus,
    TaxoDataset,
    build_train_label_vector,
    build_eval_label_vector,
    build_train_label_matrix,
    build_eval_label_matrix,
    load_class_keywords,
    DocStore,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/sagemaker-user/20252R0136DATA30400/project_release/Amazon_products",
        help="Path to Amazon_products directory",
    )
    parser.add_argument("--max_train_docs", type=int, default=200)
    parser.add_argument("--max_test_docs", type=int, default=50)
    parser.add_argument("--tok", type=str, default="bert-base-uncased")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pad_to_max_length", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1) Load taxonomy
    taxonomy = Taxonomy.load_from_dir(args.data_dir)
    print(f"[OK] taxonomy loaded: num_classes={taxonomy.num_classes}")
    print(f"     example classes: 0->{taxonomy.id2name.get(0)}, 1->{taxonomy.id2name.get(1)}")

    # 2) Load keywords (optional)
    kw_path = os.path.join(args.data_dir, "class_related_keywords.txt")
    if os.path.exists(kw_path):
        kw = load_class_keywords(kw_path)
        some_key = next(iter(kw.keys())) if len(kw) > 0 else None
        print(f"[OK] keywords loaded: {len(kw)} classes with keywords")
        if some_key is not None:
            print(f"     example: {some_key} -> {kw[some_key][:5]}")
    else:
        print("[SKIP] class_related_keywords.txt not found (ok)")

    # 3) Load corpora
    train_path = os.path.join(args.data_dir, "train", "train_corpus.txt")
    test_path = os.path.join(args.data_dir, "test", "test_corpus.txt")

    train_docs = read_corpus(train_path)[: args.max_train_docs]
    test_docs = read_corpus(test_path)[: args.max_test_docs]

    print(f"[OK] train docs loaded: {len(train_docs)} (show first 1)")
    if train_docs:
        print(f"     train[0].id={train_docs[0].doc_id}, text[:80]={train_docs[0].text[:80]!r}")
    print(f"[OK] test docs loaded : {len(test_docs)} (show first 1)")
    if test_docs:
        print(f"     test[0].id={test_docs[0].doc_id}, text[:80]={test_docs[0].text[:80]!r}")

    # 4) DocStore lookup test (doc_id != index-safe)
    store = DocStore(train_docs)
    first_id = train_docs[0].doc_id if train_docs else 0
    d0 = store.by_id(first_id)
    print("[OK] DocStore.by_id:", d0.doc_id if d0 else None)
    print("[OK] DocStore.text   :", (store.text(first_id) or "")[:60].replace("\n", " "))

    # 5) Build fake core labels (Stage2 not run yet)
    #    Intentionally create some empty cores to test drop_empty_core behavior.
    core_map = {}
    for i, d in enumerate(train_docs):
        if i % 13 == 0:
            core_map[d.doc_id] = []  # empty core
        else:
            c1 = (i * 17) % taxonomy.num_classes
            if i % 7 == 0:
                c2 = (i * 31) % taxonomy.num_classes
                core_map[d.doc_id] = [c1, c2]
            else:
                core_map[d.doc_id] = [c1]

    # 6) Train dataset smoke test (labels in {1,0,-1})
    ds_train = TaxoDataset(
        docs=train_docs,
        taxonomy=taxonomy,
        tokenizer_name_or_path=args.tok,
        max_length=args.max_len,
        mode="train",
        core_labels=core_map,
        drop_empty_core=True,
        pad_to_max_length=args.pad_to_max_length,
    )

    print(f"[OK] ds_train size after drop_empty_core=True: {len(ds_train)} (from {len(train_docs)})")

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=TaxoDataset.collate_fn,
    )

    batch = next(iter(dl_train))
    print("\n[TRAIN BATCH]")
    print(" keys:", list(batch.keys()))
    print(" input_ids:", tuple(batch["input_ids"].shape))
    print(" attention_mask:", tuple(batch["attention_mask"].shape))
    print(" labels:", tuple(batch["labels"].shape), batch["labels"].dtype)
    print(" labels_float:", tuple(batch["labels_float"].shape), batch["labels_float"].dtype)
    print(" label_mask:", tuple(batch["label_mask"].shape), batch["label_mask"].dtype)

    labels = batch["labels"]  # int8 {1,0,-1}
    n_pos = int((labels == 1).sum().item())
    n_neg = int((labels == 0).sum().item())
    n_ign = int((labels == -1).sum().item())
    print(f" label counts: pos={n_pos}, neg={n_neg}, ignore={n_ign}")

    # Mask correctness: masked positions should equal ignore positions
    masked = int((batch["label_mask"] == 0).sum().item())
    print(f" masked positions: {masked} (should equal ignore count={n_ign})")

    # 7) Vector builders directly
    sample_doc = ds_train.docs[0]
    sample_core = core_map[sample_doc.doc_id]
    y_train = build_train_label_vector(sample_core, taxonomy)
    print("\n[LABEL VECTOR CHECK]")
    print(" train y unique:", sorted(set(int(x) for x in np.unique(y_train))))
    print(" train y counts:",
          "pos", int((y_train == 1).sum()),
          "neg", int((y_train == 0).sum()),
          "ign", int((y_train == -1).sum()))

    # fake GT eval vector example (multi-label)
    fake_gt = [0, 1, 2]
    y_eval = build_eval_label_vector(fake_gt, taxonomy)
    print(" eval y unique:", sorted(set(int(x) for x in np.unique(y_eval))))
    print(" eval y ones count:", int((y_eval == 1).sum()))

    # 8) Matrix builders check
    train_mat, kept_ids = build_train_label_matrix(train_docs, core_map, taxonomy, drop_empty_core=True)
    print("\n[LABEL MATRIX CHECK]")
    print(" train_mat:", train_mat.shape, train_mat.dtype, "kept:", len(kept_ids))
    print(" train_mat unique:", sorted(set(int(x) for x in np.unique(train_mat))))

    # fake gt_map for eval matrix check
    gt_map = {d.doc_id: [0, 1] for d in test_docs}  # example only
    eval_mat, eval_ids = build_eval_label_matrix(test_docs, gt_map, taxonomy)
    print(" eval_mat:", eval_mat.shape, eval_mat.dtype, "ids:", len(eval_ids))
    print(" eval_mat unique:", sorted(set(int(x) for x in np.unique(eval_mat))))

    # 9) Inference dataset smoke test (no labels)
    ds_inf = TaxoDataset(
        docs=test_docs,
        taxonomy=taxonomy,
        tokenizer_name_or_path=args.tok,
        max_length=args.max_len,
        mode="infer",
        pad_to_max_length=args.pad_to_max_length,
    )
    dl_inf = DataLoader(ds_inf, batch_size=args.batch_size, shuffle=False, collate_fn=TaxoDataset.collate_fn)
    b2 = next(iter(dl_inf))
    print("\n[INFER BATCH]")
    print(" keys:", list(b2.keys()))
    print(" input_ids:", tuple(b2["input_ids"].shape))
    print(" attention_mask:", tuple(b2["attention_mask"].shape))
    print(" doc_id:", tuple(b2["doc_id"].shape))

    print("\n[OK] dataloader smoke test passed.")


if __name__ == "__main__":
    main()
