from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn


@dataclass(frozen=True)
class BalancedMSEConfig:
    num_bins: int = 50
    init_sigma: float = 1.0
    eps: float = 1e-8


class BalancedMSELoss(nn.Module):
    """Balanced MSE (engineering approximation described in docs/PLAN.md).

    loss = NLL(y | y_pred, sigma) + logZ(y_pred, sigma)
    where logZ = log Σ_b p_train(bin_b) * N(bin_center_b | y_pred, sigma^2).
    """

    def __init__(self, *, bin_centers: torch.Tensor, bin_probs: torch.Tensor, cfg: BalancedMSEConfig):
        super().__init__()
        if bin_centers.ndim != 1 or bin_probs.ndim != 1:
            raise ValueError("bin_centers/bin_probs must be 1D")
        if bin_centers.shape[0] != bin_probs.shape[0]:
            raise ValueError("bin_centers and bin_probs must have same length")
        self.cfg = cfg
        self.register_buffer("bin_centers", bin_centers.float())
        self.register_buffer("bin_probs", (bin_probs / bin_probs.sum()).float())
        self.log_sigma = nn.Parameter(torch.tensor(float(cfg.init_sigma)).log())

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        eps = float(self.cfg.eps)
        # BalancedMSE relies on log-sum-exp over bins; keep it in fp32 for stability even under AMP/bf16.
        device_type = "cuda" if y_pred.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            y_pred32 = y_pred.float()
            y_true32 = y_true.float()

            # Clamp sigma for numerical stability (and use the clamped value consistently).
            sigma = self.log_sigma.exp().clamp(min=0.05, max=10.0)
            log_sigma = sigma.log()
            var = sigma * sigma

            diff = y_pred32 - y_true32
            nll = (diff * diff) / (2.0 * var) + log_sigma

            # logZ: (B,) via log-sum-exp for stability.
            centers = self.bin_centers.to(y_pred32.device)
            probs = self.bin_probs.to(y_pred32.device)
            log_probs = (probs.clamp_min(eps)).log()
            dd = centers[None, :] - y_pred32[:, None]  # (B, bins)
            log_gauss = -(dd * dd) / (2.0 * var) - torch.log(sigma * math.sqrt(2.0 * math.pi))
            logZ = torch.logsumexp(log_probs[None, :] + log_gauss, dim=-1)
            return (nll + logZ).mean()


def make_histogram_bins(y: torch.Tensor, *, num_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    y = y.detach().float().cpu()
    if y.numel() == 0:
        raise ValueError("Empty labels for histogram")
    y_min = float(y.min())
    y_max = float(y.max())
    if y_max <= y_min:
        y_max = y_min + 1.0
    edges = torch.linspace(y_min, y_max, int(num_bins) + 1)
    counts = torch.histc(y, bins=int(num_bins), min=y_min, max=y_max)
    centers = 0.5 * (edges[:-1] + edges[1:])
    probs = counts / counts.sum().clamp_min(1.0)
    return centers, probs
