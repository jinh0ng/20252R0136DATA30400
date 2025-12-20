# src/submission.py
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

# Ensure src/ is on sys.path so "from utils..." works
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.paths import resolve_paths
from utils.seed import set_seed
from utils.io import ensure_dir
from utils.embeddings import load_all_embeddings, load_pid_lists


# =========================
# GAT Model (동일 구조)
# =========================
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, negative_slope=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
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

        h = self.W(x)
        h = h.view(C, self.num_heads, self.out_dim)
        h = self.dropout(h)

        h_heads = h.permute(1, 0, 2)

        alpha_src = (h_heads * self.a_src.unsqueeze(1)).sum(dim=-1)
        alpha_dst = (h_heads * self.a_dst.unsqueeze(1)).sum(dim=-1)

        e = alpha_src.unsqueeze(2) + alpha_dst.unsqueeze(1)
        e = self.leaky_relu(e)

        adj_h = adj.unsqueeze(0)
        e = e.masked_fill(adj_h == 0, float("-inf"))

        attn = F.softmax(e, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, h_heads)
        out = out.permute(1, 0, 2)

        out = out.reshape(C, self.num_heads * self.out_dim)
        return out


class TaxoClassGAT(nn.Module):
    def __init__(self, doc_dim, label_dim, hidden_dim, num_classes, adj, label_init, num_heads=4):
        super().__init__()
        self.num_classes = num_classes
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
        logits = torch.matmul(doc_h, label_h.t())
        return logits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make submission.csv using EMA teacher (notebook-equivalent).")

    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--outputs_dir", type=str, default=None)

    p.add_argument("--emb_subdir", type=str, default="embeddings_mpnet")
    p.add_argument("--pid_subdir", type=str, default="embeddings_mpnet")
    p.add_argument("--train_doc_name", type=str, default="train_doc_mpnet.npy")  # unused
    p.add_argument("--test_doc_name", type=str, default="test_doc_mpnet.npy")
    p.add_argument("--class_name_name", type=str, default="class_name_mpnet.npy")

    p.add_argument("--adj_path", type=str, default=None,
                   help="Default: <data_dir>/graph/adj.pt if exists else adj.npy")

    p.add_argument("--teacher_ckpt", type=str, default=None,
                   help="Default: <outputs_dir>/checkpoints/ema_teacher.pt")

    # dynamic topk (threshold=0.75, min 2 max 3)
    p.add_argument("--threshold", type=float, default=0.75)
    p.add_argument("--min_labels", type=int, default=2)
    p.add_argument("--max_labels", type=int, default=3)

    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=4)

    p.add_argument("--pred_batch", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--out_name", type=str, default="submission.csv")
    return p.parse_args()


def load_adj(adj_path: Path, device: torch.device) -> torch.Tensor:
    if adj_path.suffix == ".pt":
        adj = torch.load(adj_path, map_location="cpu")
        if isinstance(adj, torch.Tensor):
            return adj.float().to(device)
        raise ValueError(f"adj.pt must be a torch.Tensor, got {type(adj)}")
    elif adj_path.suffix == ".npy":
        adj_np = np.load(adj_path).astype(np.float32)
        return torch.from_numpy(adj_np).float().to(device)
    else:
        raise ValueError(f"Unsupported adj format: {adj_path}")


@torch.no_grad()
def predict_probs(model: nn.Module, X_np: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    N = X_np.shape[0]
    # infer C by one forward
    probs_all: Optional[np.ndarray] = None

    for i in tqdm(range(0, N, batch_size), desc="[submission] predict"):
        xb = torch.from_numpy(X_np[i:i + batch_size]).float().to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        if probs_all is None:
            probs_all = np.zeros((N, probs.shape[1]), dtype=np.float32)
        probs_all[i:i + probs.shape[0]] = probs

    assert probs_all is not None
    return probs_all


def dynamic_topk_labels(prob_row: np.ndarray, threshold: float, min_k: int, max_k: int) -> List[int]:
    candidates = np.where(prob_row >= threshold)[0].tolist()

    if len(candidates) < min_k:
        topk = prob_row.argsort()[-min_k:][::-1]
        candidates = topk.tolist()
    elif len(candidates) > max_k:
        # candidates 중 score 상위 max_k개
        cand_scores = prob_row[candidates]
        top_idx = np.argsort(cand_scores)[-max_k:][::-1]
        candidates = [candidates[i] for i in top_idx]

    candidates = sorted(candidates)
    return candidates


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[submission] device:", device)

    paths = resolve_paths(
        dataset_dir=args.dataset_dir,
        data_dir=args.data_dir,
        outputs_dir=args.outputs_dir,
        dataset_name="Amazon_products",
    )
    dataset_dir = paths.dataset_dir
    data_dir = paths.data_dir
    outputs_dir = paths.outputs_dir

    # adj path
    if args.adj_path:
        adj_path = Path(args.adj_path)
    else:
        adj_pt = data_dir / "graph" / "adj.pt"
        adj_npy = data_dir / "graph" / "adj.npy"
        adj_path = adj_pt if adj_pt.exists() else adj_npy
    if not adj_path.exists():
        raise FileNotFoundError(f"Missing adj: {adj_path} (run build_silver.py with --save_adj_pt or ensure adj.npy exists)")

    # teacher ckpt
    teacher_ckpt = Path(args.teacher_ckpt) if args.teacher_ckpt else (outputs_dir / "checkpoints" / "ema_teacher.pt")
    if not teacher_ckpt.exists():
        raise FileNotFoundError(f"Missing teacher_ckpt: {teacher_ckpt} (run self_train.py first)")

    # Load embeddings (need test_doc_emb + class_name_emb)
    emb = load_all_embeddings(
        dataset_dir=dataset_dir,
        emb_subdir=args.emb_subdir,
        pid_subdir=args.pid_subdir,
        train_doc_name=args.train_doc_name,
        test_doc_name=args.test_doc_name,
        class_name_name=args.class_name_name,
        strict_alignment=False,
        try_load_pid_lists=False,
    )
    test_doc_emb = emb.test_doc_emb.astype(np.float32)
    class_name_emb = emb.class_name_emb.astype(np.float32)

    # pid_list_test
    pid_train, pid_test = load_pid_lists(dataset_dir=dataset_dir, pid_subdir=args.pid_subdir)
    pid_list_test = pid_test.astype(object)

    C, label_dim = class_name_emb.shape
    doc_dim = test_doc_emb.shape[1]
    if label_dim != doc_dim:
        raise ValueError(f"Embedding dim mismatch: doc_dim={doc_dim}, label_dim={label_dim}")

    adj_tensor = load_adj(adj_path, device=device)
    if tuple(adj_tensor.shape) != (C, C):
        raise ValueError(f"adj shape mismatch: got {tuple(adj_tensor.shape)} expected {(C, C)}")

    label_init = torch.from_numpy(class_name_emb).float().to(device)

    # Build model
    model = TaxoClassGAT(
        doc_dim=doc_dim,
        label_dim=label_dim,
        hidden_dim=args.hidden_dim,
        num_classes=C,
        adj=adj_tensor,
        label_init=label_init,
        num_heads=args.num_heads,
    ).to(device)

    # Load teacher weights
    ckpt_obj = torch.load(teacher_ckpt, map_location="cpu")
    state = ckpt_obj["model_state_dict"] if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj else ckpt_obj
    model.load_state_dict(state, strict=True)
    print("[submission] loaded teacher ckpt:", teacher_ckpt)

    # Predict probs
    probs = predict_probs(model, test_doc_emb, device=device, batch_size=args.pred_batch)
    print("[submission] test_probs:", probs.shape, "| min/max:", float(probs.min()), float(probs.max()))

    # Dynamic Top2~Top3
    labels_str_list: List[str] = []
    for row in probs:
        labs = dynamic_topk_labels(
            row,
            threshold=args.threshold,
            min_k=args.min_labels,
            max_k=args.max_labels,
        )
        labels_str_list.append(",".join(str(x) for x in labs))

    # Save submission
    sub_dir = ensure_dir(outputs_dir / "submissions")
    out_path = sub_dir / args.out_name

    submission = pd.DataFrame(
        {
            "id": pid_list_test.astype(str),
            "labels": labels_str_list,
        }
    )
    submission.to_csv(out_path, index=False, encoding="utf-8")
    print("[submission] saved:", out_path)
    print(submission.head())


if __name__ == "__main__":
    main()
