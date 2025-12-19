# src/test_similarity.py
# -*- coding: utf-8 -*-

import argparse
import os
import time

from data.dataloader import Taxonomy, read_corpus
from document_similarity import DocumentSim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/sagemaker-user/20252R0136DATA30400/project_release/Amazon_products",
    )
    parser.add_argument("--model", type=str, default="roberta-large-mnli")
    parser.add_argument("--device", type=str, default=None)  # None => auto
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--num_docs", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=12)
    parser.add_argument("--cache_dir", type=str, default="outputs/stage1/sim_cache")
    args = parser.parse_args()

    # 1) load taxonomy + a few classes
    tax = Taxonomy.load_from_dir(args.data_dir)
    class_ids = list(range(min(args.num_classes, tax.num_classes)))
    class_names = [tax.id2name[i] for i in class_ids]

    # 2) load a few docs
    train_path = os.path.join(args.data_dir, "train", "train_corpus.txt")
    docs = read_corpus(train_path)[: args.num_docs]
    doc_ids = [d.doc_id for d in docs]
    doc_texts = [d.text for d in docs]

    print(f"[INFO] using {len(docs)} docs, {len(class_ids)} classes")
    print(f"[INFO] cache_dir={args.cache_dir}")

    # 3) init similarity
    sim = DocumentSim(
        model_name_or_path=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_len,
        cache_dir=args.cache_dir,
        use_amp=True,
    )

    # 4) first pass (cache miss)
    t0 = time.time()
    for did, txt in zip(doc_ids, doc_texts):
        r = sim.score_doc(did, txt, class_ids, class_names, update_cache=True)
        topk = sorted(zip(r.class_ids.tolist(), r.probs.tolist()), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n[DOC {did}] top5 entail probs:")
        for cid, p in topk:
            print(f"  class_id={cid:3d} name={tax.id2name[cid]:30s} prob={p:.4f}")
    t1 = time.time()
    print(f"\n[TIME] first pass (compute+save cache): {t1 - t0:.2f}s")

    # 5) second pass (cache hit)
    t2 = time.time()
    for did, txt in zip(doc_ids, doc_texts):
        _ = sim.score_doc(did, txt, class_ids, class_names, update_cache=True)
    t3 = time.time()
    print(f"[TIME] second pass (cache hit): {t3 - t2:.2f}s")

    # 6) verify cache files exist
    ok = True
    for did in doc_ids:
        fp = os.path.join(args.cache_dir, f"{did}.npz")
        if not os.path.exists(fp):
            ok = False
            print(f"[WARN] cache missing: {fp}")
    if ok:
        print("[OK] cache files created and reused.")


if __name__ == "__main__":
    main()
