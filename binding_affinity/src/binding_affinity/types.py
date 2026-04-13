from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass(frozen=True)
class BatchGraph:
    z: torch.LongTensor  # (N,)
    pos: torch.FloatTensor  # (N,3)
    edge_index: torch.LongTensor  # (2,E) with (src, dst)
    batch: torch.LongTensor  # (N,) in [0, B-1]
    fa_t: torch.FloatTensor  # (B,3)
    fa_R: torch.FloatTensor  # (B,4,3,3)
    edge_attr: Optional[torch.LongTensor] = None  # (E,) optional (e.g. bond_type)
    is_ligand: Optional[torch.BoolTensor] = None  # (N,)
    atom_name: Optional[torch.LongTensor] = None  # (N,) optional (protein atom name id; 0 for ligand/UNK)
    ligand_feat: Optional[torch.FloatTensor] = None  # (N,F) optional (ligand RDKit features; 0 for protein)

    def to(self, device: torch.device) -> "BatchGraph":
        return BatchGraph(
            z=self.z.to(device),
            pos=self.pos.to(device),
            edge_index=self.edge_index.to(device),
            batch=self.batch.to(device),
            fa_t=self.fa_t.to(device),
            fa_R=self.fa_R.to(device),
            edge_attr=None if self.edge_attr is None else self.edge_attr.to(device),
            is_ligand=None if self.is_ligand is None else self.is_ligand.to(device),
            atom_name=None if self.atom_name is None else self.atom_name.to(device),
            ligand_feat=None if self.ligand_feat is None else self.ligand_feat.to(device),
        )


BatchDict = Dict[str, Any]
