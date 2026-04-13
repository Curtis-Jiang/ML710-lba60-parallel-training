from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from binding_affinity.graphs.rbf import rbf_expand
from binding_affinity.models.affinity.encoder_faenet import FAENetConfig, FAENetEncoder
from binding_affinity.models.affinity.energy_head import EnergyHead, EnergyHeadConfig
from binding_affinity.models.affinity.regressor import AffinityRegressor, RegressorConfig
from binding_affinity.models.affinity.tokenizer_hier import HierTokenizer, TokenizerConfig
from binding_affinity.types import BatchGraph


@dataclass(frozen=True)
class AffinityModelConfig:
    encoder: FAENetConfig
    energy_head: EnergyHeadConfig
    tokenizer: Optional[TokenizerConfig]
    regressor: RegressorConfig


def _batched_pos_in_frame(graph: BatchGraph, frame_idx: int) -> torch.Tensor:
    # pos_centered: (N,3)
    t = graph.fa_t[graph.batch]  # (N,3)
    pos_c = graph.pos - t
    Rk = graph.fa_R[graph.batch, frame_idx]  # (N,3,3)
    return torch.einsum("ni,nij->nj", pos_c, Rk)


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size,) + src.shape[1:], dtype=src.dtype, device=src.device)
    return out.index_add(0, index, src)


class AffinityModel(nn.Module):
    def __init__(self, cfg: AffinityModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = FAENetEncoder(cfg.encoder)
        self.energy_head = EnergyHead(cfg.energy_head)
        self.tokenizer = HierTokenizer(cfg.tokenizer) if cfg.tokenizer is not None else None
        self.regressor = AffinityRegressor(cfg.regressor, enable_it=(cfg.tokenizer is not None))
        self.cross_edge_mlp: Optional[nn.Module] = None
        if bool(cfg.regressor.cross_enabled):
            d = int(cfg.encoder.d_model)
            edge_in = int(cfg.encoder.rbf_kernels) + 2 * d
            self.cross_edge_mlp = nn.Sequential(nn.Linear(edge_in, d), nn.SiLU(), nn.Linear(d, d))
            with torch.no_grad():
                last = self.cross_edge_mlp[-1]
                if isinstance(last, nn.Linear):
                    last.weight.zero_()
                    if last.bias is not None:
                        last.bias.zero_()

    def _encode_energy(self, graph: BatchGraph, *, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        E_sum = torch.zeros((batch_size,), device=graph.pos.device, dtype=graph.pos.dtype)
        h_sum = torch.zeros((graph.z.shape[0], self.cfg.encoder.d_model), device=graph.pos.device, dtype=graph.pos.dtype)
        for k in range(4):
            pos_k = _batched_pos_in_frame(graph, frame_idx=k)
            h_k = self.encoder(
                z=graph.z,
                pos=pos_k,
                edge_index=graph.edge_index,
                batch=graph.batch,
                edge_attr=graph.edge_attr,
                is_ligand=graph.is_ligand,
                atom_name=graph.atom_name,
                ligand_feat=graph.ligand_feat,
            )
            _, E_k = self.energy_head(h_k, batch=graph.batch, batch_size=batch_size)
            h_sum = h_sum + h_k
            E_sum = E_sum + E_k
        return h_sum / 4.0, E_sum / 4.0

    def _pool_cross(self, *, h: torch.Tensor, graph: BatchGraph, batch_size: int) -> torch.Tensor:
        if self.cross_edge_mlp is None:
            raise RuntimeError("cross_edge_mlp is disabled")
        if graph.is_ligand is None:
            raise ValueError("complex graph missing is_ligand")
        if graph.edge_index.numel() == 0:
            return torch.zeros((batch_size, int(self.cfg.encoder.d_model)), device=h.device, dtype=h.dtype)

        src = graph.edge_index[0]
        dst = graph.edge_index[1]
        cross = graph.is_ligand[src] != graph.is_ligand[dst]
        if not bool(cross.any()):
            return torch.zeros((batch_size, int(self.cfg.encoder.d_model)), device=h.device, dtype=h.dtype)

        src = src[cross]
        dst = dst[cross]
        dist = torch.linalg.norm(graph.pos[src] - graph.pos[dst], dim=-1)
        rbf = rbf_expand(dist, num_kernels=int(self.cfg.encoder.rbf_kernels), cutoff=float(self.cfg.encoder.cutoff))
        edge_in = torch.cat([rbf, h[src], h[dst]], dim=-1)
        edge_h = self.cross_edge_mlp(edge_in)
        edge_batch = graph.batch[src].long()
        pooled = _scatter_sum(edge_h, edge_batch, dim_size=int(batch_size))
        ones = torch.ones((edge_batch.shape[0], 1), device=h.device, dtype=h.dtype)
        count = _scatter_sum(ones, edge_batch, dim_size=int(batch_size)).clamp_min(1.0)
        return pooled / count

    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        complex_g: BatchGraph = batch["complex"]
        protein_g: BatchGraph = batch["protein"]
        ligand_g: BatchGraph = batch["ligand"]
        y = batch.get("y")

        batch_size = int(y.shape[0]) if y is not None else int(complex_g.fa_t.shape[0])

        h_c, E_c = self._encode_energy(complex_g, batch_size=batch_size)
        _, E_p = self._encode_energy(protein_g, batch_size=batch_size)
        _, E_l = self._encode_energy(ligand_g, batch_size=batch_size)
        dE = E_c - E_p - E_l

        IT = None
        aux: Dict[str, Any] = {}
        if self.tokenizer is not None:
            if complex_g.is_ligand is None:
                raise ValueError("complex graph missing is_ligand")
            IT, aux_tok = self.tokenizer(
                h=h_c,
                pos=complex_g.pos,
                is_ligand=complex_g.is_ligand,
                batch=complex_g.batch,
                batch_size=batch_size,
            )
            aux.update(aux_tok)

        cross_vec = None
        if self.cross_edge_mlp is not None:
            cross_vec = self._pool_cross(h=h_c, graph=complex_g, batch_size=batch_size)
        y_pred = self.regressor(dE, IT, cross_vec=cross_vec)
        debug = {"dE": dE.detach(), "E_complex": E_c.detach(), "E_protein": E_p.detach(), "E_ligand": E_l.detach(), **aux}
        return y_pred, debug
