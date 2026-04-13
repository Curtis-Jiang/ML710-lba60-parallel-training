from __future__ import annotations

from torch import Tensor, nn

from sequence_binding.models.common import (
    DualEncoderClassifier,
    FusionClassifier,
    TokenPositionEmbedding,
)


class AttentionBranchEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        pad_id: int,
        max_length: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ff_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = TokenPositionEmbedding(
            vocab_size=vocab_size,
            max_length=max_length,
            hidden_dim=hidden_dim,
            pad_id=pad_id,
            dropout=dropout,
        )
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * ff_multiplier,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        hidden = self.embedding(tokens)
        key_padding_mask = ~mask
        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask=key_padding_mask)
        hidden = self.norm(hidden)
        return hidden * mask.unsqueeze(-1).to(hidden.dtype)


def build_attention_model(
    *,
    protein_vocab_size: int,
    smiles_vocab_size: int,
    protein_pad_id: int,
    smiles_pad_id: int,
    protein_max_length: int,
    smiles_max_length: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    ff_multiplier: int,
    dropout: float,
) -> DualEncoderClassifier:
    protein_encoder = AttentionBranchEncoder(
        vocab_size=protein_vocab_size,
        pad_id=protein_pad_id,
        max_length=protein_max_length,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_multiplier=ff_multiplier,
        dropout=dropout,
    )
    smiles_encoder = AttentionBranchEncoder(
        vocab_size=smiles_vocab_size,
        pad_id=smiles_pad_id,
        max_length=smiles_max_length,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_multiplier=ff_multiplier,
        dropout=dropout,
    )
    classifier = FusionClassifier(hidden_dim=hidden_dim, dropout=dropout)
    return DualEncoderClassifier(
        protein_encoder=protein_encoder,
        smiles_encoder=smiles_encoder,
        classifier=classifier,
        model_name="attention",
    )
