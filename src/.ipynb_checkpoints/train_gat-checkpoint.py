# src/train_gat.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# Ensure src/ is on sys.path so "from utils..." works
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.paths import resolve_paths
from utils.seed import set_seed
from utils.io import ensure_dir, save_json
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
# GAT Model (노트북과 동일)
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

        # 1) linear
        h = self.W(x)  # (C, H*F)
        h = h.view(C, self.num_heads, self.out_dim)  # (C, H, F)
        h = self.dropout(h)

        # 2) attention score
        h_heads = h.permute(1, 0, 2)  # (H, C, F)

        alpha_src = (h_heads * self.a_src.unsqueeze(1)).sum(dim=-1)  # (H, C)
        alpha_dst = (h_heads * self.a_dst.unsqueeze(1)).sum(dim=-1)  # (H, C)

        e = alpha_src.unsqueeze(2) + alpha_dst.unsqueeze(1)  # (H, C, C)
        e = self.leaky_relu(e)

        # 3) mask with adjacency
        adj_h = adj.unsqueeze(0)  # (1, C, C)
        e = e.masked_fill(adj_h == 0, float("-inf"))

        # 4) softmax
        attn = F.softmax(e, dim=-1)  # (H, C, C)
        attn = self.dropout(attn)

        # 5) weighted sum
        out = torch.matmul(attn, h_heads)  # (H, C, F)
        out = out.permute(1, 0, 2)  # (C, H, F)

        out = out.reshape(C, self.num_heads * self.out_dim)  # (C, H*F)
        return out


class TaxoClassGAT(nn.Module):
    def __init__(self, doc_dim, label_dim, hidden_dim, num_classes,
                 adj, label_init, num_heads=4):
        super().__init__()
        self.num_classes = num_classes
        self.adj = adj                  # (C, C)
        self.label_init = label_init    # (C, label_dim)

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
        h = self.gat1(self.label_init, self.adj)  # (C, hidden_dim)
        h = F.elu(h)
        h = self.gat2(h, self.adj)               # (C, hidden_dim)
        return h

    def forward(self, doc_x):
        doc_h = self.doc_proj(doc_x)               # (N, hidden_dim)
        label_h = self.encode_labels()             # (C, hidden_dim)
        logits = torch.matmul(doc_h, label_h.t())  # (N, C)
        return logits


# =========================
# CLI / Main
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TaxoClassGAT (notebook-equivalent).")

    p.add_argument("--dataset_dir", type=str, required=True,
                   help="~/new/project_release/Amazon_products")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Default: <repo>/data (we will read y_silver/adj here)")
    p.add_argument("--outputs_dir", type=str, default=None,
                   help="Default: <repo>/outputs")

    # embeddings
    p.add_argument("--emb_subdir", type=str, default="embeddings_mpnet")
    p.add_argument("--pid_subdir", type=str, default="embeddings_mpnet")
    p.add_argument("--train_doc_name", type=str, default="train_doc_mpnet.npy")
    p.add_argument("--test_doc_name", type=str, default="test_doc_mpnet.npy")  # unused but loaded for completeness
    p.add_argument("--class_name_name", type=str, default="class_name_mpnet.npy")

    # silver/graph (from build_silver outputs)
    p.add_argument("--y_silver_path", type=str, default=None,
                   help="Default: <data_dir>/silver/y_silver.npy")
    p.add_argument("--adj_path", type=str, default=None,
                   help="Default: <data_dir>/graph/adj.npy")

    # notebook-fixed hyperparams (keep defaults identical)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)

    # device
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cpu. Default: auto")

    # output names
    p.add_argument("--ckpt_name", type=str, default="gat.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[train_gat] device:", device)

    paths = resolve_paths(
        dataset_dir=args.dataset_dir,
        data_dir=args.data_dir,
        outputs_dir=args.outputs_dir,
        dataset_name="Amazon_products",
    )
    dataset_dir = paths.dataset_dir
    data_dir = paths.data_dir
    outputs_dir = paths.outputs_dir

    print("[train_gat] dataset_dir:", dataset_dir)
    print("[train_gat] data_dir   :", data_dir)
    print("[train_gat] outputs_dir:", outputs_dir)

    # Resolve inputs
    y_silver_path = Path(args.y_silver_path) if args.y_silver_path else (data_dir / "silver" / "y_silver.npy")
    adj_path = Path(args.adj_path) if args.adj_path else (data_dir / "graph" / "adj.npy")

    if not y_silver_path.exists():
        raise FileNotFoundError(f"Missing y_silver: {y_silver_path} (run build_silver.py first)")
    if not adj_path.exists():
        raise FileNotFoundError(f"Missing adj: {adj_path} (run build_silver.py first)")

    # Load embeddings (MPNet)
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

    # Load silver labels / adjacency
    Y_silver = np.load(y_silver_path).astype(np.float32)  # (N, C)
    adj = np.load(adj_path).astype(np.float32)            # (C, C)

    # Sanity checks
    N, doc_dim = train_doc_emb.shape
    C, label_dim = class_name_emb.shape
    if Y_silver.shape != (N, C):
        raise ValueError(f"Y_silver shape mismatch: got {Y_silver.shape}, expected {(N, C)}")
    if adj.shape != (C, C):
        raise ValueError(f"adj shape mismatch: got {adj.shape}, expected {(C, C)}")

    print("[train_gat] train_doc_emb:", train_doc_emb.shape)
    print("[train_gat] class_name_emb:", class_name_emb.shape)
    print("[train_gat] Y_silver:", Y_silver.shape, "| avg positives/doc:", float(Y_silver.sum(axis=1).mean()))
    print("[train_gat] adj:", adj.shape, "| nnz:", int(adj.sum()))

    # Torch tensors for model
    adj_tensor = torch.from_numpy(adj).float().to(device)
    label_init = torch.from_numpy(class_name_emb).float().to(device)

    # Dataset / split (노트북과 동일: 90/10)
    full_dataset = EmbeddingDataset(train_doc_emb, Y_silver)
    train_size = int(0.9 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # Model (노트북과 동일)
    model_gat = TaxoClassGAT(
        doc_dim=doc_dim,
        label_dim=label_dim,
        hidden_dim=args.hidden_dim,
        num_classes=C,
        adj=adj_tensor,
        label_init=label_init,
        num_heads=args.num_heads,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model_gat.parameters(), lr=args.lr)

    print(model_gat)

    # Train loop (노트북과 동일한 로직)
    EPOCHS = args.epochs
    train_losses: List[float] = []
    valid_losses: List[float] = []

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model_gat.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model_gat(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        avg_train = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train)

        # ---- valid ----
        model_gat.eval()
        total_v = 0.0
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model_gat(batch_x)
                loss = criterion(logits, batch_y)
                total_v += loss.item() * batch_x.size(0)

        avg_valid = total_v / len(valid_loader.dataset)
        valid_losses.append(avg_valid)

        print(f"[GAT Epoch {epoch:02d}] train={avg_train:.4f} | valid={avg_valid:.4f}")

    # Outputs
    ckpt_dir = ensure_dir(outputs_dir / "checkpoints")
    metrics_dir = ensure_dir(outputs_dir / "metrics")
    plots_dir = ensure_dir(outputs_dir / "plots")

    ckpt_path = ckpt_dir / args.ckpt_name
    torch.save(
        {
            "model_state_dict": model_gat.state_dict(),
            "config": {
                "doc_dim": doc_dim,
                "label_dim": label_dim,
                "hidden_dim": args.hidden_dim,
                "num_heads": args.num_heads,
                "num_classes": C,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "seed": args.seed,
                "dataset_dir": str(dataset_dir),
                "data_dir": str(data_dir),
                "emb_subdir": args.emb_subdir,
                "train_doc_name": args.train_doc_name,
                "class_name_name": args.class_name_name,
                "y_silver_path": str(y_silver_path),
                "adj_path": str(adj_path),
            },
            "train_losses": train_losses,
            "valid_losses": valid_losses,
        },
        ckpt_path,
    )
    print("[train_gat] saved checkpoint:", ckpt_path)

    # Save losses json
    losses_json = {
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "lr": args.lr,
    }
    save_json(metrics_dir / "gat_losses.json", losses_json)
    print("[train_gat] saved losses:", metrics_dir / "gat_losses.json")

    # Loss plot (노트북 plot과 동일한 정보)
    try:
        import matplotlib.pyplot as plt

        epochs = list(range(1, EPOCHS + 1))
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_losses, marker="o", label="Train loss")
        plt.plot(epochs, valid_losses, marker="o", label="Valid loss")
        plt.xlabel("Epoch")
        plt.ylabel("BCEWithLogits Loss")
        plt.title("Training / Validation Loss")
        plt.legend()
        plt.grid(True)

        fig_path = plots_dir / "gat_loss_curve.png"
        plt.savefig(fig_path, dpi=160, bbox_inches="tight")
        plt.close()
        print("[train_gat] saved plot:", fig_path)
    except Exception as e:
        print("[train_gat] plot skipped (matplotlib not available or failed):", repr(e))

    print("[train_gat] DONE.")


if __name__ == "__main__":
    main()
