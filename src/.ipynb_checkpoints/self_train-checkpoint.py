# src/self_train.py
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# Ensure src/ is on sys.path so "from utils..." works
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.paths import resolve_paths
from utils.seed import set_seed
from utils.io import ensure_dir, save_json, save_npy
from utils.embeddings import load_all_embeddings


# =========================
# Dataset
# =========================
class EmbeddingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0]
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# GAT Model 
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
        """
        x: (C, in_dim)
        adj: (C, C) 0/1 adjacency (self-loop 포함)
        """
        C = x.size(0)

        h = self.W(x)  # (C, H*F)
        h = h.view(C, self.num_heads, self.out_dim)  # (C, H, F)
        h = self.dropout(h)

        h_heads = h.permute(1, 0, 2)  # (H, C, F)

        alpha_src = (h_heads * self.a_src.unsqueeze(1)).sum(dim=-1)  # (H, C)
        alpha_dst = (h_heads * self.a_dst.unsqueeze(1)).sum(dim=-1)  # (H, C)

        e = alpha_src.unsqueeze(2) + alpha_dst.unsqueeze(1)  # (H, C, C)
        e = self.leaky_relu(e)

        adj_h = adj.unsqueeze(0)  # (1, C, C)
        e = e.masked_fill(adj_h == 0, float("-inf"))

        attn = F.softmax(e, dim=-1)  # (H, C, C)
        attn = self.dropout(attn)

        out = torch.matmul(attn, h_heads)  # (H, C, F)
        out = out.permute(1, 0, 2)  # (C, H, F)

        out = out.reshape(C, self.num_heads * self.out_dim)  # (C, H*F)
        return out


class TaxoClassGAT(nn.Module):
    def __init__(self, doc_dim, label_dim, hidden_dim, num_classes, adj, label_init, num_heads=4):
        super().__init__()
        self.num_classes = num_classes
        self.adj = adj
        self.label_init = label_init

        head_out = hidden_dim // num_heads
        assert head_out * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

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


# =========================
# EMA + Pseudo label refine
# =========================
def update_ema_model(ema_model: nn.Module, student_model: nn.Module, alpha: float = 0.99) -> None:
    """
    EMA 업데이트: θ_te ← α θ_te + (1 − α) θ
    """
    for p_te, p in zip(ema_model.parameters(), student_model.parameters()):
        p_te.data.mul_(alpha).add_(p.data, alpha=1.0 - alpha)


@torch.no_grad()
def refine_labels_with_ema_teacher(
    ema_model: nn.Module,
    X_np: np.ndarray,
    Y_old: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
    t_pos: float = 0.9,
    t_neg: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    EMA teacher로 pseudo-label 정제:
    - prob >= t_pos → 1로 강제 (확실한 positive)
    - prob <= t_neg 이면서 기존이 1 → 0으로 (확실한 negative)
    ( GPU 메모리 위해 batch inference)
    """
    ema_model.eval()

    N = X_np.shape[0]
    C = Y_old.shape[1]
    probs_all = np.zeros((N, C), dtype=np.float32)

    for i in tqdm(range(0, N, batch_size), desc="[self_train] EMA predict"):
        xb = torch.from_numpy(X_np[i:i + batch_size]).float().to(device)
        logits = ema_model(xb)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        probs_all[i:i + probs.shape[0]] = probs

    Y_new = Y_old.copy()

    pos_mask = probs_all >= t_pos
    Y_new[pos_mask] = 1.0

    neg_mask = probs_all <= t_neg
    drop_mask = (Y_new == 1.0) & neg_mask
    Y_new[drop_mask] = 0.0

    print("Refinement summary (EMA):")
    print("  newly set to 1:", int(pos_mask.sum()))
    print("  dropped from 1:", int(drop_mask.sum()))

    return Y_new, probs_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EMA self-training for TaxoClassGAT (notebook-equivalent).")

    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--outputs_dir", type=str, default=None)

    # embeddings
    p.add_argument("--emb_subdir", type=str, default="embeddings_mpnet")
    p.add_argument("--pid_subdir", type=str, default="embeddings_mpnet")
    p.add_argument("--train_doc_name", type=str, default="train_doc_mpnet.npy")
    p.add_argument("--test_doc_name", type=str, default="test_doc_mpnet.npy")  # unused here
    p.add_argument("--class_name_name", type=str, default="class_name_mpnet.npy")

    # inputs from build_silver
    p.add_argument("--y_silver_path", type=str, default=None,
                   help="Default: <data_dir>/silver/y_silver.npy")
    p.add_argument("--adj_path", type=str, default=None,
                   help="Default: <data_dir>/graph/adj.pt if exists else adj.npy")

    # starting checkpoint (student init)
    p.add_argument("--student_ckpt", type=str, default=None,
                   help="Default: <outputs_dir>/checkpoints/gat.pt")

    # notebook hyperparams
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=256)          # training batch (BATCH_SIZE)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)

    # EMA self-training params 
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--epochs_st", type=int, default=3)
    p.add_argument("--alpha_ema", type=float, default=0.99)
    p.add_argument("--t_pos", type=float, default=0.9)
    p.add_argument("--t_neg", type=float, default=0.1)

    # EMA inference batch (여기선 배치)
    p.add_argument("--ema_pred_batch", type=int, default=1024)

    # device
    p.add_argument("--device", type=str, default=None)

    return p.parse_args()


def load_adj(adj_path: Path, device: torch.device) -> torch.Tensor:
    """
    Prefer .pt if provided; else load .npy.
    """
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[self_train] device:", device)

    paths = resolve_paths(
        dataset_dir=args.dataset_dir,
        data_dir=args.data_dir,
        outputs_dir=args.outputs_dir,
        dataset_name="Amazon_products",
    )
    dataset_dir = paths.dataset_dir
    data_dir = paths.data_dir
    outputs_dir = paths.outputs_dir

    print("[self_train] dataset_dir:", dataset_dir)
    print("[self_train] data_dir   :", data_dir)
    print("[self_train] outputs_dir:", outputs_dir)

    # paths
    y_silver_path = Path(args.y_silver_path) if args.y_silver_path else (data_dir / "silver" / "y_silver.npy")
    if not y_silver_path.exists():
        raise FileNotFoundError(f"Missing y_silver: {y_silver_path} (run build_silver.py first)")

    # prefer adj.pt if not specified and exists
    if args.adj_path:
        adj_path = Path(args.adj_path)
    else:
        adj_pt = data_dir / "graph" / "adj.pt"
        adj_npy = data_dir / "graph" / "adj.npy"
        adj_path = adj_pt if adj_pt.exists() else adj_npy
    if not adj_path.exists():
        raise FileNotFoundError(f"Missing adj: {adj_path} (run build_silver.py with --save_adj_pt or ensure adj.npy exists)")

    # starting student ckpt
    student_ckpt = Path(args.student_ckpt) if args.student_ckpt else (outputs_dir / "checkpoints" / "gat.pt")
    if not student_ckpt.exists():
        raise FileNotFoundError(f"Missing student checkpoint: {student_ckpt} (run train_gat.py first)")

    # Load embeddings
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
    train_doc_emb = emb.train_doc_emb.astype(np.float32)  # (N, 768)
    class_name_emb = emb.class_name_emb.astype(np.float32)  # (C, 768)

    # Load labels/graph
    Y_silver = np.load(y_silver_path).astype(np.float32)  # (N, C)
    adj_tensor = load_adj(adj_path, device=device)

    N, doc_dim = train_doc_emb.shape
    C, label_dim = class_name_emb.shape
    if Y_silver.shape != (N, C):
        raise ValueError(f"Y_silver shape mismatch: got {Y_silver.shape}, expected {(N, C)}")
    if tuple(adj_tensor.shape) != (C, C):
        raise ValueError(f"adj shape mismatch: got {tuple(adj_tensor.shape)}, expected {(C, C)}")

    label_init = torch.from_numpy(class_name_emb).float().to(device)

    # Build student model
    student = TaxoClassGAT(
        doc_dim=doc_dim,
        label_dim=label_dim,
        hidden_dim=args.hidden_dim,
        num_classes=C,
        adj=adj_tensor,
        label_init=label_init,
        num_heads=args.num_heads,
    ).to(device)

    # Load checkpoint weights
    ckpt_obj = torch.load(student_ckpt, map_location="cpu")
    state = ckpt_obj["model_state_dict"] if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj else ckpt_obj
    student.load_state_dict(state, strict=True)
    print("[self_train] loaded student ckpt:", student_ckpt)

    # Init ema teacher as deep copy of student
    ema_teacher = copy.deepcopy(student).to(device)
    ema_teacher.eval()
    print("[self_train] initialized EMA teacher from student")

    criterion = nn.BCEWithLogitsLoss()

    # current labels
    Y_current = Y_silver.copy()

    ckpt_dir = ensure_dir(outputs_dir / "checkpoints")
    metrics_dir = ensure_dir(outputs_dir / "metrics")
    silver_dir = ensure_dir(data_dir / "silver")  # store refined labels per round

    history = {
        "student_ckpt": str(student_ckpt),
        "y_silver_path": str(y_silver_path),
        "adj_path": str(adj_path),
        "rounds": args.rounds,
        "epochs_st": args.epochs_st,
        "alpha_ema": args.alpha_ema,
        "t_pos": args.t_pos,
        "t_neg": args.t_neg,
        "ema_pred_batch": args.ema_pred_batch,
        "train_losses": [],
        "valid_losses": [],
    }

    for r in range(1, args.rounds + 1):
        print(f"\n========== EMA SELF-TRAINING ROUND {r} ==========")

        # 1) refine pseudo labels with teacher
        Y_refined, _probs = refine_labels_with_ema_teacher(
            ema_model=ema_teacher,
            X_np=train_doc_emb,
            Y_old=Y_current,
            device=device,
            batch_size=args.ema_pred_batch,
            t_pos=args.t_pos,
            t_neg=args.t_neg,
        )

        # optional: save refined label
        y_round_path = silver_dir / f"y_refined_round{r}.npy"
        save_npy(y_round_path, Y_refined)
        print("[self_train] saved refined labels:", y_round_path)

        # 2) dataset split (90/10)
        dataset_refined = EmbeddingDataset(train_doc_emb, Y_refined)
        train_size = int(0.9 * len(dataset_refined))
        valid_size = len(dataset_refined) - train_size
        train_ds, valid_ds = random_split(dataset_refined, [train_size, valid_size])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

        # 3) student fine-tune + EMA update (step마다)
        optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs_st + 1):
            student.train()
            total_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                logits = student(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                # EMA update
                update_ema_model(ema_teacher, student, alpha=args.alpha_ema)

                total_loss += loss.item() * batch_x.size(0)

            avg_train = total_loss / len(train_loader.dataset)

            # validation
            student.eval()
            total_v = 0.0
            with torch.no_grad():
                for batch_x, batch_y in valid_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    logits = student(batch_x)
                    loss = criterion(logits, batch_y)
                    total_v += loss.item() * batch_x.size(0)

            avg_valid = total_v / len(valid_loader.dataset)

            print(f"[EMA Round {r} / Epoch {epoch}] train={avg_train:.4f} | valid={avg_valid:.4f}")
            history["train_losses"].append({"round": r, "epoch": epoch, "loss": float(avg_train)})
            history["valid_losses"].append({"round": r, "epoch": epoch, "loss": float(avg_valid)})

        # round end: update current labels
        Y_current = Y_refined

        # save checkpoints per round
        student_path = ckpt_dir / f"gat_student_round{r}.pt"
        teacher_path = ckpt_dir / f"gat_ema_teacher_round{r}.pt"

        torch.save({"model_state_dict": student.state_dict()}, student_path)
        torch.save({"model_state_dict": ema_teacher.state_dict()}, teacher_path)

        print("[self_train] saved student ckpt:", student_path)
        print("[self_train] saved teacher ckpt:", teacher_path)

    # final teacher checkpoint name (easy for submission.py)
    final_teacher = ckpt_dir / "ema_teacher.pt"
    torch.save({"model_state_dict": ema_teacher.state_dict()}, final_teacher)
    print("[self_train] saved FINAL teacher ckpt:", final_teacher)

    # meta save
    meta_path = metrics_dir / "self_train_meta.json"
    save_json(meta_path, history)
    print("[self_train] saved meta:", meta_path)

    print("[self_train] DONE.")


if __name__ == "__main__":
    main()
