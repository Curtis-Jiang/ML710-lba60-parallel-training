from __future__ import annotations

import torch


def rbf_expand(dist: torch.Tensor, *, num_kernels: int, cutoff: float, eps: float = 1e-8) -> torch.Tensor:
    """Gaussian RBF expansion for distances.

    Args:
      dist: (...,) distances (float).
    Returns:
      (..., num_kernels) float tensor.
    """
    dist = dist.clamp_min(0.0)
    centers = torch.linspace(0.0, float(cutoff), int(num_kernels), device=dist.device, dtype=dist.dtype)
    widths = (centers[1] - centers[0]).clamp_min(eps)
    gamma = 1.0 / (widths * widths)
    diff = dist.unsqueeze(-1) - centers
    return torch.exp(-gamma * diff * diff)

