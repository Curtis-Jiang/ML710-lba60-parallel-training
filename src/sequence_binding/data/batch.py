from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PairBatch:
    protein_tokens: Tensor
    protein_mask: Tensor
    smiles_tokens: Tensor
    smiles_mask: Tensor
    labels: Tensor

    def to(self, device: torch.device) -> "PairBatch":
        return PairBatch(
            protein_tokens=self.protein_tokens.to(device, non_blocking=True),
            protein_mask=self.protein_mask.to(device, non_blocking=True),
            smiles_tokens=self.smiles_tokens.to(device, non_blocking=True),
            smiles_mask=self.smiles_mask.to(device, non_blocking=True),
            labels=self.labels.to(device, non_blocking=True),
        )
