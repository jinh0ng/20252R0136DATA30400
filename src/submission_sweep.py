# src/submission_sweep.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.paths import resolve_paths
from utils.seed import set_seed
from utils.io import ensure_dir
from utils.embeddings import load_all_embeddings, load_pid_lists


# ---------- Model (same as train_gat/submission.py) ----------
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, negative_slope=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(num_heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(num_heads, out_dim))

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, adj):
        C = x.size(0)
        h = self.W(x).view(C, self.num_heads, self.out_dim)
        h = self.dropout(h)

        h_heads = h.permute(1, 0, 2)  # (H,C,F)
        alpha_src = (h_heads * self.a_src.unsqueeze(1)).sum(dim=-1)  # (H,C)
        alpha_dst = (h_heads * self.a_dst.unsqueeze(1)).sum(dim=-1)  # (H,C)

        e = self.leaky_relu(alpha_src.unsqueeze(2) + alpha_dst.unsqueeze(1))  # (H,C,C)
        e = e.masked_fill(adj.unsqueeze(0) == 0, float("-inf"))

        attn = self.dropout(F.softmax(e, dim=-1))
        out = torch.matmul(attn, h_heads).permute(1, 0, 2).reshape(C, -1)
        return out


class TaxoClassGAT(nn.Module):
    def __init__(self, doc_dim, label_dim, hidden_dim, num_classes, adj, label_init, num_heads=4):
        super().__init__()
        self.adj = adj
        self.label_init = label_init

        head_out = hidden_dim // num_heads
        assert head_out * num_heads == hidden_dim

        self.gat1 = GATLayer(label_dim, head_out, num_heads=num_heads, dropout=0.1)
        self.gat2 = GATLayer(hidden_dim, head_out, num_heads=num_heads, dropout=0.1)

        self.doc_proj = nn.Sequential(
            nn.Linear(doc_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def encode_labels(self):
        h = self.gat1(self.label_init, self.adj)
        h = F.elu(h)
        h = self.gat2(h, self.adj)
        return h

    def forward(self, doc_x):
        doc_h = self.doc_proj(doc_x)
        label_h = self.encode_labels()
        return torch.matmul(doc_h, label_h.t())


def load_adj(adj_path: Path, device: torch.device) -> torch.Tensor:
    if adj_path.suffix == ".pt":
        adj = torch.load(adj_path, map_location="cpu")
        if not isinstance(adj, torch.Tensor):
            raise ValueError(f"adj.pt must be torch.Tensor, got {type(adj)}")
        return adj.float().to(device)
    adj_np = np.load(adj_path).astype(np.float32)
    return torch.from_numpy(adj_np).float().to(device)


@torch.no_grad()
def predict_probs(model: nn.Module, X_np: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    N = X_np.shape[0]
    probs_all = None

    for i in tqdm(range(0, N, batch_size), desc="[sweep] predict"):
        xb = torch.from_numpy(X_np[i:i + batch_size]).float().to(device)
        probs = torch.sigmoid(model(xb)).cpu().numpy().astype(np.float32)
        if probs_all is None:
            probs_all = np.zeros((N, probs.shape[1]), dtype=np.float32)
        probs_all[i:i + probs.shape[0]] = probs

    assert probs_all is not None
    return probs_all


def dynamic_topk(prob_row: np.ndarray, threshold: float, min_k: int, max_k: int) -> List[int]:
    cand = np.where(prob_row >= threshold)[0].tolist()
    if len(cand) < min_k:
        cand = prob_row.argsort()[-min_k:][::-1].tolist()
    elif len(cand) > max_k:
        scores = prob_row[cand]
        top_idx = np.argsort(scores)[-max_k:][::-1]
        cand = [cand[i] for i in top_idx]
    return sorted(cand)


def fixed_topk(prob_row: np.ndarray, k: int) -> List[int]:
    return sorted(prob_row.argsort()[-k:][::-1].tolist())


def parse_args():
    p = argparse.ArgumentParser("submission sweep")
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--outputs_dir", default=None)

    p.add_argument("--emb_subdir", default="embeddings_mpnet")
    p.add_argument("--pid_subdir", default="embeddings_mpnet")
    p.add_argument("--test_doc_name", default="test_doc_mpnet.npy")
    p.add_argument("--class_name_name", default="class_name_mpnet.npy")

    p.add_argument("--teacher_ckpt", default=None,
                   help="Default: <outputs_dir>/checkpoints/ema_teacher.pt")
    p.add_argument("--adj_path", default=None,
                   help="Default: <data_dir>/graph/adj.pt if exists else adj.npy")

    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=4)

    p.add_argument("--pred_batch", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)

    # sweep settings
    p.add_argument("--thresholds", type=str, default="0.50,0.55,0.60,0.62,0.65,0.68,0.70,0.72,0.75")
    p.add_argument("--min_labels", type=int, default=2)
    p.add_argument("--max_labels", type=int, default=3)
    p.add_argument("--also_fixed", action="store_true",
                   help="Also produce fixed top2/top3 submissions")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[sweep] device:", device)

    paths = resolve_paths(args.dataset_dir, args.data_dir, args.outputs_dir, dataset_name="Amazon_products")
    dataset_dir, data_dir, outputs_dir = paths.dataset_dir, paths.data_dir, paths.outputs_dir

    # adj path
    if args.adj_path:
        adj_path = Path(args.adj_path)
    else:
        adj_pt = data_dir / "graph" / "adj.pt"
        adj_npy = data_dir / "graph" / "adj.npy"
        adj_path = adj_pt if adj_pt.exists() else adj_npy
    if not adj_path.exists():
        raise FileNotFoundError(f"Missing adj: {adj_path}")

    # teacher ckpt
    teacher_ckpt = Path(args.teacher_ckpt) if args.teacher_ckpt else (outputs_dir / "checkpoints" / "ema_teacher.pt")
    if not teacher_ckpt.exists():
        raise FileNotFoundError(f"Missing teacher ckpt: {teacher_ckpt}")

    # load embeddings
    emb = load_all_embeddings(
        dataset_dir=dataset_dir,
        emb_subdir=args.emb_subdir,
        pid_subdir=args.pid_subdir,
        train_doc_name="train_doc_mpnet.npy",  # unused
        test_doc_name=args.test_doc_name,
        class_name_name=args.class_name_name,
        strict_alignment=False,
        try_load_pid_lists=False,
    )
    test_doc = emb.test_doc_emb.astype(np.float32)
    class_name = emb.class_name_emb.astype(np.float32)

    _, pid_test = load_pid_lists(dataset_dir=dataset_dir, pid_subdir=args.pid_subdir)
    pid_test = pid_test.astype(str)

    C, label_dim = class_name.shape
    doc_dim = test_doc.shape[1]

    adj = load_adj(adj_path, device)
    label_init = torch.from_numpy(class_name).float().to(device)

    model = TaxoClassGAT(
        doc_dim=doc_dim,
        label_dim=label_dim,
        hidden_dim=args.hidden_dim,
        num_classes=C,
        adj=adj,
        label_init=label_init,
        num_heads=args.num_heads,
    ).to(device)

    ckpt = torch.load(teacher_ckpt, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    probs = predict_probs(model, test_doc, device=device, batch_size=args.pred_batch)

    sub_dir = ensure_dir(outputs_dir / "submissions")

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    for th in thresholds:
        labels = []
        for row in probs:
            labs = dynamic_topk(row, threshold=th, min_k=args.min_labels, max_k=args.max_labels)
            labels.append(",".join(map(str, labs)))

        df = pd.DataFrame({"id": pid_test, "labels": labels})
        out_path = sub_dir / f"submission_dyn_th{th:.2f}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print("[sweep] saved:", out_path)

    if args.also_fixed:
        for k in [2, 3]:
            labels = []
            for row in probs:
                labs = fixed_topk(row, k=k)
                labels.append(",".join(map(str, labs)))
            df = pd.DataFrame({"id": pid_test, "labels": labels})
            out_path = sub_dir / f"submission_top{k}.csv"
            df.to_csv(out_path, index=False, encoding="utf-8")
            print("[sweep] saved:", out_path)

    print("[sweep] DONE. Check:", sub_dir)


if __name__ == "__main__":
    main()
