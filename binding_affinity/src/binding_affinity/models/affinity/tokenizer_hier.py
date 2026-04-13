from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from binding_affinity.graphs.rbf import rbf_expand


@dataclass(frozen=True)
class TokenizerConfig:
    d_model: int = 128
    kp: int = 192
    kl: int = 32
    kint: int = 64
    attn_heads: int = 4
    dropout: float = 0.0
    bias_rbf_kernels: int = 16
    bias_cutoff: float = 20.0
    eps: float = 1e-8


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HierTokenizer(nn.Module):
    """Hierarchical soft clustering + cross-attention to produce interaction tokens."""

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        if cfg.d_model % cfg.attn_heads != 0:
            raise ValueError("d_model must be divisible by attn_heads")
        self.cfg = cfg

        self.assign_p = _MLP(cfg.d_model, cfg.d_model, cfg.kp, dropout=cfg.dropout)
        self.assign_l = _MLP(cfg.d_model, cfg.d_model, cfg.kl, dropout=cfg.dropout)

        self.w_q = nn.Linear(cfg.d_model, cfg.d_model)
        self.w_k = nn.Linear(cfg.d_model, cfg.d_model)
        self.w_v = nn.Linear(cfg.d_model, cfg.d_model)
        self.w_o = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln = nn.LayerNorm(cfg.d_model)

        self.bias_mlp = _MLP(cfg.bias_rbf_kernels, cfg.d_model, 1, dropout=cfg.dropout)
        self.assign_int = _MLP(cfg.d_model, cfg.d_model, cfg.kint, dropout=cfg.dropout)

    def _clusters(
        self, H: torch.Tensor, X: torch.Tensor, *, assign_mlp: nn.Module, K: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (A, C, G, loc_loss)."""
        eps = self.cfg.eps
        if H.numel() == 0:
            A = torch.zeros((0, K), device=H.device, dtype=H.dtype)
            C = torch.zeros((K, self.cfg.d_model), device=H.device, dtype=H.dtype)
            G = torch.zeros((K, 3), device=H.device, dtype=X.dtype)
            return A, C, G, torch.zeros((), device=H.device, dtype=H.dtype)

        A = F.softmax(assign_mlp(H), dim=-1)  # (N,K)
        AT = A.transpose(0, 1)  # (K,N)
        denom = AT.sum(dim=1, keepdim=True).clamp_min(eps)  # (K,1)
        C = (AT @ H) / denom
        C = F.normalize(C, dim=-1)
        G = (AT @ X) / denom.to(X.dtype)

        # locality: weighted variance around centroid
        diff = X[:, None, :] - G[None, :, :]
        dist2 = (diff * diff).sum(dim=-1)  # (N,K)
        numer = (A * dist2).sum(dim=0)  # (K,)
        loc = (numer / (A.sum(dim=0).clamp_min(eps))).sum()
        return A, C, G, loc

    def _entropy(self, A: torch.Tensor) -> torch.Tensor:
        eps = self.cfg.eps
        if A.numel() == 0:
            return torch.zeros((), device=A.device, dtype=A.dtype)
        ent = -(A * (A.clamp_min(eps).log())).sum(dim=-1).mean()
        return ent

    def _cross_attn(self, Cl: torch.Tensor, Cp: torch.Tensor, Gl: torch.Tensor, Gp: torch.Tensor) -> torch.Tensor:
        if Cl.numel() == 0 or Cp.numel() == 0:
            return Cl

        cfg = self.cfg
        h = int(cfg.attn_heads)
        d = int(cfg.d_model)
        dh = d // h

        Q = self.w_q(Cl).view(cfg.kl, h, dh).transpose(0, 1)  # (h,Kl,dh)
        K = self.w_k(Cp).view(cfg.kp, h, dh).transpose(0, 1)  # (h,Kp,dh)
        V = self.w_v(Cp).view(cfg.kp, h, dh).transpose(0, 1)  # (h,Kp,dh)

        logits = torch.matmul(Q, K.transpose(-2, -1)) / (dh**0.5)  # (h,Kl,Kp)
        dist = torch.cdist(Gl, Gp)  # (Kl,Kp)
        bias = self.bias_mlp(rbf_expand(dist, num_kernels=cfg.bias_rbf_kernels, cutoff=cfg.bias_cutoff)).squeeze(-1)
        logits = logits + bias.unsqueeze(0)

        attn = F.softmax(logits, dim=-1)
        out = torch.matmul(attn, V)  # (h,Kl,dh)
        out = out.transpose(0, 1).contiguous().view(cfg.kl, d)
        out = self.w_o(out)
        return self.ln(Cl + out)

    def forward(
        self,
        *,
        h: torch.Tensor,
        pos: torch.Tensor,
        is_ligand: torch.BoolTensor,
        batch: torch.LongTensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Tokenization is numerically sensitive; force fp32 even under AMP/bf16.
        device_type = "cuda" if h.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            h32 = h.float()
            pos32 = pos.float()
            cfg = self.cfg
            IT_all = []
            loc_all = []
            ent_all = []

            for b in range(int(batch_size)):
                mask_b = batch == b
                hb = h32[mask_b]
                xb = pos32[mask_b]
                lb = is_ligand[mask_b]

                Hp = hb[~lb]
                Hl = hb[lb]
                Xp = xb[~lb]
                Xl = xb[lb]

                Ap, Cp, Gp, loc_p = self._clusters(Hp, Xp, assign_mlp=self.assign_p, K=cfg.kp)
                Al, Cl, Gl, loc_l = self._clusters(Hl, Xl, assign_mlp=self.assign_l, K=cfg.kl)
                loc = loc_p + loc_l

                ent = self._entropy(Ap) + self._entropy(Al)
                Clp = self._cross_attn(Cl, Cp, Gl, Gp)

                Aint = F.softmax(self.assign_int(Clp), dim=-1) if Clp.numel() else torch.zeros((0, cfg.kint), device=h.device, dtype=torch.float32)
                IT = (Aint.transpose(0, 1) @ Clp) if Clp.numel() else torch.zeros((cfg.kint, cfg.d_model), device=h.device, dtype=torch.float32)
                IT_all.append(IT)
                loc_all.append(loc)
                ent_all.append(ent)

            IT_batch = (
                torch.stack(IT_all, dim=0) if IT_all else torch.zeros((0, cfg.kint, cfg.d_model), device=h.device, dtype=torch.float32)
            )
            aux = {
                "loc": torch.stack(loc_all).mean() if loc_all else torch.zeros((), device=h.device, dtype=torch.float32),
                "entropy": torch.stack(ent_all).mean() if ent_all else torch.zeros((), device=h.device, dtype=torch.float32),
            }
        return IT_batch.to(h.dtype), {k: v.to(h.dtype) for k, v in aux.items()}
