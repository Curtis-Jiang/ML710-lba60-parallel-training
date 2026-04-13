from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from sequence_binding.data.batch import PairBatch


def masked_mean(hidden: Tensor, mask: Tensor) -> Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denom


class TokenPositionEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        hidden_dim: int,
        pad_id: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim,
            padding_idx=pad_id,
        )
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_dim**0.5

    def forward(self, tokens: Tensor) -> Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) * self.scale
        hidden = hidden + self.position_embedding(positions)
        return self.dropout(hidden)


class FusionClassifier(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, protein_repr: Tensor, smiles_repr: Tensor) -> Tensor:
        features = torch.cat(
            [
                protein_repr,
                smiles_repr,
                (protein_repr - smiles_repr).abs(),
                protein_repr * smiles_repr,
            ],
            dim=-1,
        )
        return self.net(features).squeeze(-1)


@dataclass
class ModelMetadata:
    name: str
    parameter_count: int
    mamba_backend: str | None = None


class DualEncoderClassifier(nn.Module):
    def __init__(
        self,
        *,
        protein_encoder: nn.Module,
        smiles_encoder: nn.Module,
        classifier: FusionClassifier,
        model_name: str,
        mamba_backend: str | None = None,
    ) -> None:
        super().__init__()
        self.protein_encoder = protein_encoder
        self.smiles_encoder = smiles_encoder
        self.classifier = classifier
        self.model_name = model_name
        self.mamba_backend = mamba_backend

    def forward(self, batch: PairBatch) -> Tensor:
        protein_hidden = self.protein_encoder(batch.protein_tokens, batch.protein_mask)
        smiles_hidden = self.smiles_encoder(batch.smiles_tokens, batch.smiles_mask)
        protein_repr = masked_mean(protein_hidden, batch.protein_mask)
        smiles_repr = masked_mean(smiles_hidden, batch.smiles_mask)
        return self.classifier(protein_repr, smiles_repr)

    def metadata(self) -> ModelMetadata:
        parameter_count = sum(param.numel() for param in self.parameters())
        return ModelMetadata(
            name=self.model_name,
            parameter_count=int(parameter_count),
            mamba_backend=self.mamba_backend,
        )
