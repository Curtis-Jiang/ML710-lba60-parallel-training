from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from binding_affinity.graphs.rbf import rbf_expand


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size,) + src.shape[1:], dtype=src.dtype, device=src.device)
    return out.index_add(0, index, src)


class MLP(nn.Module):
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


@dataclass(frozen=True)
class FAENetConfig:
    d_model: int = 128
    layers: int = 4
    rbf_kernels: int = 16
    cutoff: float = 5.0
    intra_cutoff: Optional[float] = None
    use_ligand_flag: bool = True
    # How to embed node ids:
    # - "combined": embed z directly (vocab size = z_vocab)
    # - "factorized": interpret z as z_id = atomic_number + 128 * res_id, then
    #   embed element and residue separately and sum.
    z_embedding: str = "combined"
    z_vocab: int = 128
    edge_swish: bool = False
    use_bond_emb: bool = False
    graph_norm: bool = False
    graph_norm_eps: float = 1e-5
    # Edge-type modeling:
    # - "shared": shared edge MLPs (current behavior)
    # - "type_emb": add per-edge-type embedding to edge features (pp/ll/cross)
    # - "split_mlp": split filter MLPs for intra vs cross edges
    edge_type_mode: str = "shared"
    # Interaction topology:
    # - "single_stream": mix intra+cross messages per layer (current behavior)
    # - "two_stream": intra update then cross update per layer
    interaction_mode: str = "single_stream"
    # Distance envelope:
    # - "none": no envelope (current behavior)
    # - "cosine": cosine cutoff envelope applied to edge features
    envelope: str = "none"
    # Optional node-level chemical semantics:
    use_atom_name_emb: bool = False
    atom_name_vocab: int = 64
    ligand_feat_dim: int = 0


class FAENetEncoder(nn.Module):
    """IPBind/FAENet-style message passing (node features only)."""

    def __init__(self, cfg: FAENetConfig):
        super().__init__()
        self.cfg = cfg
        self.ligand_feat_dim = int(cfg.ligand_feat_dim)

        self.graph_norm: Optional[_GraphNorm] = None
        if bool(cfg.graph_norm):
            self.graph_norm = _GraphNorm(cfg.d_model, eps=float(cfg.graph_norm_eps))

        emb_mode = str(cfg.z_embedding or "combined").lower()
        self.z_emb: Optional[nn.Embedding] = None
        self.elem_emb: Optional[nn.Embedding] = None
        self.res_emb: Optional[nn.Embedding] = None
        if emb_mode == "combined":
            self.z_emb = nn.Embedding(int(cfg.z_vocab), cfg.d_model)
        elif emb_mode == "factorized":
            self.elem_emb = nn.Embedding(128, cfg.d_model)
            # Residue id is packed into z via: z_id = atomic_number + 128 * res_id.
            # Reserve extra slots for UNK/variants; 0 is used for UNK / ligand.
            self.res_emb = nn.Embedding(32, cfg.d_model)
            with torch.no_grad():
                self.res_emb.weight[0].zero_()
        else:
            raise ValueError(f"Unknown z_embedding mode: {cfg.z_embedding!r}")
        self.lig_emb = nn.Embedding(2, cfg.d_model) if cfg.use_ligand_flag else None
        self.bond_emb = nn.Embedding(4, cfg.d_model) if bool(cfg.use_bond_emb) else None
        if self.bond_emb is not None:
            with torch.no_grad():
                self.bond_emb.weight[0].zero_()

        self.atom_name_emb: Optional[nn.Embedding] = None
        self.ligand_feat_mlp: Optional[MLP] = None

        self.edge_type_mode = str(cfg.edge_type_mode or "shared").lower()
        if self.edge_type_mode not in {"shared", "type_emb", "split_mlp"}:
            raise ValueError(f"Unknown edge_type_mode: {cfg.edge_type_mode!r}")
        self.interaction_mode = str(cfg.interaction_mode or "single_stream").lower()
        if self.interaction_mode not in {"single_stream", "two_stream"}:
            raise ValueError(f"Unknown interaction_mode: {cfg.interaction_mode!r}")
        self.envelope = str(cfg.envelope or "none").lower()
        if self.envelope not in {"none", "cosine"}:
            raise ValueError(f"Unknown envelope: {cfg.envelope!r}")

        self.edge_type_emb: Optional[nn.Embedding] = None
        if self.edge_type_mode == "type_emb":
            self.edge_type_emb = nn.Embedding(3, cfg.d_model)
            with torch.no_grad():
                self.edge_type_emb.weight.zero_()

        edge_in = int(cfg.rbf_kernels) + 3
        edge_dim = cfg.d_model
        self.edge_mlp = MLP(edge_in, cfg.d_model, edge_dim)

        self.filter_mlps: Optional[nn.ModuleList] = None
        self.filter_mlps_intra: Optional[nn.ModuleList] = None
        self.filter_mlps_cross: Optional[nn.ModuleList] = None
        if self.edge_type_mode == "split_mlp":
            self.filter_mlps_intra = nn.ModuleList(
                [MLP(edge_dim + 2 * cfg.d_model, 2 * cfg.d_model, cfg.d_model) for _ in range(int(cfg.layers))]
            )
            self.filter_mlps_cross = nn.ModuleList(
                [MLP(edge_dim + 2 * cfg.d_model, 2 * cfg.d_model, cfg.d_model) for _ in range(int(cfg.layers))]
            )
        else:
            self.filter_mlps = nn.ModuleList(
                [MLP(edge_dim + 2 * cfg.d_model, 2 * cfg.d_model, cfg.d_model) for _ in range(int(cfg.layers))]
            )
        self.update_mlps = nn.ModuleList(
            [MLP(cfg.d_model, 2 * cfg.d_model, cfg.d_model) for _ in range(int(cfg.layers))]
        )

        if bool(cfg.use_atom_name_emb):
            self.atom_name_emb = nn.Embedding(int(cfg.atom_name_vocab), cfg.d_model)
            with torch.no_grad():
                self.atom_name_emb.weight[0].zero_()

        if self.ligand_feat_dim > 0:
            self.ligand_feat_mlp = MLP(self.ligand_feat_dim, cfg.d_model, cfg.d_model)
            with torch.no_grad():
                last = self.ligand_feat_mlp.net[-1]
                if isinstance(last, nn.Linear):
                    last.weight.zero_()
                    if last.bias is not None:
                        last.bias.zero_()

    def forward(
        self,
        *,
        z: torch.LongTensor,
        pos: torch.Tensor,
        edge_index: torch.LongTensor,
        batch: Optional[torch.LongTensor] = None,
        edge_attr: Optional[torch.LongTensor] = None,
        is_ligand: Optional[torch.BoolTensor] = None,
        atom_name: Optional[torch.LongTensor] = None,
        ligand_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if z.numel() == 0:
            return torch.zeros((0, self.cfg.d_model), device=pos.device, dtype=pos.dtype)

        if self.z_emb is not None:
            zc = z.clamp_min(0).clamp_max(int(self.cfg.z_vocab) - 1)
            h = self.z_emb(zc)
        else:
            assert self.elem_emb is not None and self.res_emb is not None
            zc = z.clamp_min(0)
            z_base = torch.remainder(zc, 128).clamp_max(127)
            res_id = torch.div(zc, 128, rounding_mode="floor").clamp_max(31)
            h = self.elem_emb(z_base) + self.res_emb(res_id)
        if self.lig_emb is not None and is_ligand is not None:
            h = h + self.lig_emb(is_ligand.long())

        if self.atom_name_emb is not None and atom_name is not None:
            aid = atom_name.long().clamp_min(0).clamp_max(int(self.atom_name_emb.num_embeddings) - 1)
            h = h + self.atom_name_emb(aid)

        if self.ligand_feat_mlp is not None and ligand_feat is not None:
            lf = ligand_feat.float()
            if lf.ndim == 1:
                lf = lf.unsqueeze(-1)
            if lf.ndim != 2 or lf.shape[0] != h.shape[0] or lf.shape[1] != self.ligand_feat_dim:
                raise ValueError(f"ligand_feat must be (N,{self.ligand_feat_dim}): got {tuple(lf.shape)} for N={int(h.shape[0])}")
            add = self.ligand_feat_mlp(lf)
            if is_ligand is not None:
                add = add * is_ligand.float().unsqueeze(-1)
            h = h + add

        if edge_index.numel() == 0:
            return h

        src = edge_index[0]
        dst = edge_index[1]
        bond_type = edge_attr.long().view(-1) if edge_attr is not None else None
        r = pos[src] - pos[dst]
        dist = torch.linalg.norm(r, dim=-1)

        if self.cfg.intra_cutoff is not None:
            intra = float(self.cfg.intra_cutoff)
            if is_ligand is not None:
                same_mol = is_ligand[src] == is_ligand[dst]
                keep = (~same_mol) | (dist <= intra)
            else:
                keep = dist <= intra

            if keep.numel() and (not bool(torch.all(keep))):
                src = src[keep]
                dst = dst[keep]
                if bond_type is not None:
                    bond_type = bond_type[keep]
                r = r[keep]
                dist = dist[keep]
                if src.numel() == 0:
                    return h

        rbf = rbf_expand(dist, num_kernels=self.cfg.rbf_kernels, cutoff=self.cfg.cutoff)
        e = self.edge_mlp(torch.cat([rbf, r], dim=-1))
        if bool(self.cfg.edge_swish):
            e = F.silu(e)
        if self.bond_emb is not None and bond_type is not None:
            bt = bond_type.clamp_min(0).clamp_max(int(self.bond_emb.num_embeddings) - 1)
            e = e + self.bond_emb(bt)
        if self.envelope == "cosine":
            if is_ligand is not None and self.cfg.intra_cutoff is not None:
                same_mol = is_ligand[src] == is_ligand[dst]
                cut = torch.where(same_mol, torch.full_like(dist, float(self.cfg.intra_cutoff)), torch.full_like(dist, float(self.cfg.cutoff)))
            else:
                cut = torch.full_like(dist, float(self.cfg.cutoff))
            t = dist / (cut + 1e-12)
            w = 0.5 * (torch.cos(math.pi * t.clamp_max(1.0)) + 1.0)
            w = torch.where(t <= 1.0, w, torch.zeros_like(w))
            e = e * w.unsqueeze(-1)
        if self.edge_type_emb is not None and is_ligand is not None:
            src_l = is_ligand[src]
            dst_l = is_ligand[dst]
            edge_type = torch.zeros_like(src, dtype=torch.long)
            edge_type = torch.where(src_l & dst_l, torch.ones_like(edge_type), edge_type)
            edge_type = torch.where(src_l != dst_l, torch.full_like(edge_type, 2), edge_type)
            e = e + self.edge_type_emb(edge_type.long().clamp_min(0).clamp_max(2))

        n = int(h.shape[0])
        edge_is_cross = is_ligand[src] != is_ligand[dst] if is_ligand is not None else None
        if self.edge_type_mode == "split_mlp" and edge_is_cross is not None:
            assert self.filter_mlps_intra is not None and self.filter_mlps_cross is not None
            use_split = True
        else:
            use_split = False

        for layer_idx, upd in enumerate(self.update_mlps):
            if edge_is_cross is not None and self.interaction_mode == "two_stream":
                cross = edge_is_cross

                if bool((~cross).any()):
                    h_src = h[src]
                    h_dst = h[dst]
                    if use_split:
                        filt_intra = self.filter_mlps_intra[layer_idx]  # type: ignore[index]
                    else:
                        assert self.filter_mlps is not None
                        filt_intra = self.filter_mlps[layer_idx]
                    f_intra = F.silu(filt_intra(torch.cat([e[~cross], h_dst[~cross], h_src[~cross]], dim=-1)))
                    m_intra = _scatter_sum(h_src[~cross] * f_intra, dst[~cross], dim_size=n)
                    h = h + 0.5 * upd(m_intra)
                    if self.graph_norm is not None and batch is not None:
                        h = self.graph_norm(h, batch)

                if bool(cross.any()):
                    h_src = h[src]
                    h_dst = h[dst]
                    if use_split:
                        filt_cross = self.filter_mlps_cross[layer_idx]  # type: ignore[index]
                    else:
                        assert self.filter_mlps is not None
                        filt_cross = self.filter_mlps[layer_idx]
                    f_cross = F.silu(filt_cross(torch.cat([e[cross], h_dst[cross], h_src[cross]], dim=-1)))
                    m_cross = _scatter_sum(h_src[cross] * f_cross, dst[cross], dim_size=n)
                    h = h + 0.5 * upd(m_cross)
                    if self.graph_norm is not None and batch is not None:
                        h = self.graph_norm(h, batch)
                continue

            h_src = h[src]
            h_dst = h[dst]
            if use_split:
                assert edge_is_cross is not None
                m = torch.zeros((n, int(self.cfg.d_model)), device=h.device, dtype=h.dtype)
                cross = edge_is_cross
                if bool((~cross).any()):
                    filt_intra = self.filter_mlps_intra[layer_idx]  # type: ignore[index]
                    f_intra = F.silu(filt_intra(torch.cat([e[~cross], h_dst[~cross], h_src[~cross]], dim=-1)))
                    m = m + _scatter_sum(h_src[~cross] * f_intra, dst[~cross], dim_size=n)
                if bool(cross.any()):
                    filt_cross = self.filter_mlps_cross[layer_idx]  # type: ignore[index]
                    f_cross = F.silu(filt_cross(torch.cat([e[cross], h_dst[cross], h_src[cross]], dim=-1)))
                    m = m + _scatter_sum(h_src[cross] * f_cross, dst[cross], dim_size=n)
            else:
                assert self.filter_mlps is not None
                filt = self.filter_mlps[layer_idx]
                f = F.silu(filt(torch.cat([e, h_dst, h_src], dim=-1)))
                m = _scatter_sum(h_src * f, dst, dim_size=n)
            h = h + upd(m)
            if self.graph_norm is not None and batch is not None:
                h = self.graph_norm(h, batch)
        return h


class _GraphNorm(nn.Module):
    def __init__(self, d_model: int, *, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((int(d_model),), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros((int(d_model),), dtype=torch.float32))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, batch: torch.LongTensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        if batch.numel() == 0:
            return x

        b = batch.long()
        B = int(b.max().item()) + 1
        if B <= 0:
            return x

        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            x32 = x.float()
            ones = torch.ones((x32.shape[0], 1), device=x32.device, dtype=x32.dtype)
            count = _scatter_sum(ones, b, dim_size=B).clamp_min(1.0)  # (B,1)
            mean = _scatter_sum(x32, b, dim_size=B) / count  # (B,D)
            xc = x32 - mean[b]
            var = _scatter_sum(xc * xc, b, dim_size=B) / count  # (B,D)
            std = torch.sqrt(var + float(self.eps))
            xn = xc / std[b]
            out = xn * self.weight[None, :] + self.bias[None, :]
        return out.to(dtype=x.dtype)
