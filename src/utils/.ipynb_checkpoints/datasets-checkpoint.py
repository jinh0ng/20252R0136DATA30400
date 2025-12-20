# src/utils/models_gat.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Dense-adjacency GAT layer (multi-head), matching your notebook implementation:
      - linear projection W
      - additive attention: a_src·h_i + a_dst·h_j
      - adjacency mask (0 -> -inf)
      - softmax over neighbors
      - head concat
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        # W: (in_dim -> out_dim * num_heads)
        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)

        # attention vectors per head: (H, F)
        self.a_src = nn.Parameter(torch.empty(num_heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(num_heads, out_dim))

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (C, in_dim)
            adj: (C, C) with 0/1 entries (self-loop optional but recommended)
        Returns:
            out: (C, num_heads*out_dim)
        """
        C = x.size(0)
        if adj.dim() != 2 or adj.size(0) != C or adj.size(1) != C:
            raise ValueError(f"adj must be (C,C). Got {tuple(adj.shape)} for C={C}")

        # 1) linear projection
        h = self.W(x)                         # (C, H*F)
        h = h.view(C, self.num_heads, self.out_dim)  # (C, H, F)
        h = self.dropout(h)

        # 2) compute attention scores
        h_heads = h.permute(1, 0, 2)          # (H, C, F)

        alpha_src = (h_heads * self.a_src.unsqueeze(1)).sum(dim=-1)  # (H, C)
        alpha_dst = (h_heads * self.a_dst.unsqueeze(1)).sum(dim=-1)  # (H, C)

        e = alpha_src.unsqueeze(2) + alpha_dst.unsqueeze(1)          # (H, C, C)
        e = self.leaky_relu(e)

        # 3) mask by adjacency
        adj_h = adj.unsqueeze(0)                                     # (1, C, C)
        e = e.masked_fill(adj_h == 0, float("-inf"))

        # 4) softmax over neighbors
        attn = F.softmax(e, dim=-1)                                  # (H, C, C)
        attn = self.dropout(attn)

        # 5) weighted sum
        out = torch.matmul(attn, h_heads)                            # (H, C, F)
        out = out.permute(1, 0, 2).contiguous()                      # (C, H, F)
        out = out.view(C, self.num_heads * self.out_dim)             # (C, H*F)

        return out


class TaxoClassGAT(nn.Module):
    """
    TaxoClass-style model:
      - label encoder: 2-layer GAT over label graph
      - document projection: MLP
      - logits = doc_h @ label_h^T
    """
    def __init__(
        self,
        doc_dim: int,
        label_dim: int,
        hidden_dim: int,
        num_classes: int,
        adj: torch.Tensor,
        label_init: torch.Tensor,
        num_heads: int = 4,
        dropout_gat: float = 0.1,
        dropout_doc: float = 0.3,
        trainable_label_init: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        # Store graph / label init as buffers or parameters
        # - adj should be non-trainable
        self.register_buffer("adj", adj.float())

        if trainable_label_init:
            self.label_init = nn.Parameter(label_init.float())
        else:
            self.register_buffer("label_init", label_init.float())

        head_out = hidden_dim // num_heads

        self.gat1 = GATLayer(label_dim, head_out, num_heads=num_heads, dropout=dropout_gat)
        self.gat2 = GATLayer(hidden_dim, head_out, num_heads=num_heads, dropout=dropout_gat)

        self.doc_proj = nn.Sequential(
            nn.Linear(doc_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_doc),
        )

    def encode_labels(self) -> torch.Tensor:
        h = self.gat1(self.label_init, self.adj)  # (C, hidden_dim)
        h = F.elu(h)
        h = self.gat2(h, self.adj)                # (C, hidden_dim)
        return h

    def forward(self, doc_x: torch.Tensor) -> torch.Tensor:
        """
        doc_x: (N, doc_dim)
        return: logits (N, C)
        """
        doc_h = self.doc_proj(doc_x)              # (N, hidden_dim)
        label_h = self.encode_labels()            # (C, hidden_dim)
        logits = torch.matmul(doc_h, label_h.t()) # (N, C)
        return logits
