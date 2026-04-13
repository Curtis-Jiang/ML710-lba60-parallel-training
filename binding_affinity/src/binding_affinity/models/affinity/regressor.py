from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RegressorConfig:
    d_model: int = 128
    hidden: int = 256
    dropout: float = 0.0
    cross_enabled: bool = False


class AffinityRegressor(nn.Module):
    def __init__(self, cfg: RegressorConfig, *, enable_it: bool = True):
        super().__init__()
        self.cfg = cfg
        self.enable_it = bool(enable_it)
        self.de_linear = nn.Linear(1, 1)
        self.mlp: Optional[nn.Module] = None
        if self.enable_it:
            self.mlp = nn.Sequential(
                nn.Linear(cfg.d_model, int(cfg.hidden)),
                nn.SiLU(),
                nn.Dropout(float(cfg.dropout)),
                nn.Linear(int(cfg.hidden), 1),
            )
        self.cross_mlp: Optional[nn.Module] = None
        if bool(cfg.cross_enabled):
            self.cross_mlp = nn.Sequential(
                nn.Linear(cfg.d_model, int(cfg.hidden)),
                nn.SiLU(),
                nn.Dropout(float(cfg.dropout)),
                nn.Linear(int(cfg.hidden), 1),
            )
            with torch.no_grad():
                last = self.cross_mlp[-1]
                if isinstance(last, nn.Linear):
                    last.weight.zero_()
                    if last.bias is not None:
                        last.bias.zero_()

    def forward(self, dE: torch.Tensor, IT: Optional[torch.Tensor] = None, *, cross_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        y_de = self.de_linear(dE.view(-1, 1)).squeeze(-1)
        y = y_de
        if self.enable_it and (IT is not None) and (self.mlp is not None):
            pooled = IT.mean(dim=1)
            y = y + self.mlp(pooled).squeeze(-1)
        if (self.cross_mlp is not None) and (cross_vec is not None):
            y = y + self.cross_mlp(cross_vec).squeeze(-1)
        return y
