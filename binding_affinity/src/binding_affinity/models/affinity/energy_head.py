from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size,) + src.shape[1:], dtype=src.dtype, device=src.device)
    return out.index_add(0, index, src)


@dataclass(frozen=True)
class EnergyHeadConfig:
    d_model: int = 128
    hidden: int = 128


class EnergyHead(nn.Module):
    def __init__(self, cfg: EnergyHeadConfig):
        super().__init__()
        self.cfg = cfg
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, int(cfg.hidden)),
            nn.SiLU(),
            nn.Linear(int(cfg.hidden), 1),
        )

    def forward(self, h: torch.Tensor, *, batch: torch.LongTensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.mlp(h).squeeze(-1)
        E = _scatter_sum(v, batch, dim_size=int(batch_size))
        return v, E

