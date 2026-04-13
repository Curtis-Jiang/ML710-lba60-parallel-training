from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PairwiseRankingConfig:
    margin: float = 0.0


def pairwise_ranking_loss(y_pred: torch.Tensor, y_true: torch.Tensor, *, cfg: PairwiseRankingConfig) -> torch.Tensor:
    """Batch-wise pairwise ranking loss.

    For pairs (i,j) with y_i > y_j, encourage ŷ_i > ŷ_j via softplus.
    """
    if y_true.numel() <= 1:
        return torch.zeros((), device=y_true.device, dtype=y_true.dtype)

    diff = y_true[:, None] - y_true[None, :]
    mask = diff > 0
    if not torch.any(mask):
        return torch.zeros((), device=y_true.device, dtype=y_true.dtype)

    pred_diff = y_pred[:, None] - y_pred[None, :]
    loss = F.softplus(float(cfg.margin) - pred_diff)
    return loss[mask].mean()


@dataclass(frozen=True)
class ListwiseNDCGConfig:
    """Differentiable listwise NDCG loss via soft-rank approximation.

    This is an engineering approximation intended to align with the affinity-model plan:
    - use a listwise objective (vs. pairwise)
    - apply exponential gain to emphasize high-affinity examples
    """

    temperature: float = 1.0
    exp_gain_scale: float = 1.0
    eps: float = 1e-8


def _soft_rank(scores: torch.Tensor, *, temperature: float, eps: float) -> torch.Tensor:
    """Approximate ranks with pairwise sigmoids.

    rank_i ≈ 1 + Σ_j sigmoid((s_j - s_i) / tau)
    """
    tau = float(temperature)
    if tau <= 0:
        tau = 1.0
    s = scores
    diff = (s[None, :] - s[:, None]) / tau
    p = torch.sigmoid(diff)
    # Remove self-comparisons (sigmoid(0)=0.5 would bias ranks).
    eye = torch.eye(int(s.shape[0]), device=s.device, dtype=s.dtype)
    p = p * (1.0 - eye)
    return 1.0 + p.sum(dim=-1)


def listwise_ndcg_loss(y_pred: torch.Tensor, y_true: torch.Tensor, *, cfg: ListwiseNDCGConfig) -> torch.Tensor:
    """Listwise NDCG loss (1 - NDCG) with soft ranks.

    - gains use exp scaling to upweight high y_true (as mentioned in the plan).
    - discounts use log2(1 + rank_pred).
    """
    if y_true.numel() <= 1:
        return torch.zeros((), device=y_true.device, dtype=y_true.dtype)

    device_type = "cuda" if y_pred.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        y_pred32 = y_pred.float()
        y_true32 = y_true.float()
        eps = float(cfg.eps)

        # Exponential gains (stable via subtracting max; constant factor cancels in NDCG ratio).
        scale = float(cfg.exp_gain_scale)
        gains = torch.exp(scale * (y_true32 - y_true32.max()))

        ranks = _soft_rank(y_pred32, temperature=float(cfg.temperature), eps=eps)
        discounts = 1.0 / torch.log2(1.0 + ranks).clamp_min(eps)
        dcg = (gains * discounts).sum()

        # Ideal DCG from true ordering (non-differentiable but constant w.r.t. y_pred).
        y_sorted, _ = torch.sort(y_true32, descending=True)
        gains_sorted = torch.exp(scale * (y_sorted - y_sorted[0]))
        ideal_ranks = torch.arange(1, int(y_true32.shape[0]) + 1, device=y_true32.device, dtype=y_true32.dtype)
        ideal_discounts = 1.0 / torch.log2(1.0 + ideal_ranks).clamp_min(eps)
        idcg = (gains_sorted * ideal_discounts).sum()

        ndcg = dcg / (idcg + eps)
        return 1.0 - ndcg
