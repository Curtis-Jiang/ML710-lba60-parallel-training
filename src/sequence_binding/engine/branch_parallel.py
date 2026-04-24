"""Branch model-parallel wrapper: one GPU per encoder branch, fuse on primary."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from sequence_binding.data.batch import PairBatch
from sequence_binding.engine.distributed import (
    DistributedContext,
    branch_secondary_device,
)
from sequence_binding.models.common import (
    DualEncoderClassifier,
    ModelMetadata,
    masked_mean,
)


class BranchParallelWrapper(nn.Module):
    """Protein branch on ``cuda:0``, SMILES branch on ``cuda:1``, fusion on ``cuda:0``."""

    def __init__(
        self,
        inner: DualEncoderClassifier,
        primary: torch.device,
        secondary: torch.device,
    ) -> None:
        super().__init__()
        self.primary = primary
        self.secondary = secondary
        self.protein_encoder = inner.protein_encoder.to(primary)
        self.smiles_encoder = inner.smiles_encoder.to(secondary)
        self.classifier = inner.classifier.to(primary)
        self.model_name = inner.model_name
        self.mamba_backend = inner.mamba_backend

    def forward(self, batch: PairBatch) -> Tensor:
        protein_tokens = batch.protein_tokens.to(self.primary, non_blocking=True)
        protein_mask = batch.protein_mask.to(self.primary, non_blocking=True)
        p_hidden = self.protein_encoder(protein_tokens, protein_mask)
        p_repr = masked_mean(p_hidden, protein_mask)

        smiles_tokens = batch.smiles_tokens.to(self.secondary, non_blocking=True)
        smiles_mask = batch.smiles_mask.to(self.secondary, non_blocking=True)
        s_hidden = self.smiles_encoder(smiles_tokens, smiles_mask)
        s_repr = masked_mean(s_hidden, smiles_mask).to(self.primary, non_blocking=True)

        return self.classifier(p_repr, s_repr)

    def metadata(self) -> ModelMetadata:
        total = sum(param.numel() for param in self.parameters())
        return ModelMetadata(
            name=self.model_name,
            parameter_count=int(total),
            mamba_backend=self.mamba_backend,
        )


def wrap_branch_parallel(
    model: DualEncoderClassifier,
    context: DistributedContext,
) -> nn.Module:
    return BranchParallelWrapper(
        inner=model,
        primary=context.device,
        secondary=branch_secondary_device(),
    )
