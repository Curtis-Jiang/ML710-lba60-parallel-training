from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sequence_binding.models.common import (
    DualEncoderClassifier,
    FusionClassifier,
    TokenPositionEmbedding,
)

try:
    from mamba_ssm import Mamba as MambaSSM  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    MambaSSM = None


class FallbackMambaBlock(nn.Module):
    """Lightweight gated sequence block used when mamba-ssm is unavailable."""

    def __init__(self, hidden_dim: int, ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.depthwise_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_multiplier, hidden_dim),
        )

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        residual = hidden
        normalized = self.norm(hidden)
        gate, value = self.in_proj(normalized).chunk(2, dim=-1)
        value = self.depthwise_conv(value.transpose(1, 2)).transpose(1, 2)
        mixed = self.out_proj(F.silu(value) * torch.sigmoid(gate))
        hidden = residual + self.dropout(mixed * mask.unsqueeze(-1).to(mixed.dtype))
        hidden = hidden + self.dropout(
            self.ff(self.ff_norm(hidden)) * mask.unsqueeze(-1).to(hidden.dtype)
        )
        return hidden


class OfficialMambaBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ff_multiplier: int,
        dropout: float,
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> None:
        super().__init__()
        if MambaSSM is None:
            raise RuntimeError("mamba-ssm is not available")
        self.norm = nn.LayerNorm(hidden_dim)
        self.block = MambaSSM(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_multiplier, hidden_dim),
        )

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        residual = hidden
        normalized = self.norm(hidden)
        mixed = self.block(normalized)
        hidden = residual + self.dropout(mixed * mask.unsqueeze(-1).to(mixed.dtype))
        hidden = hidden + self.dropout(
            self.ff(self.ff_norm(hidden)) * mask.unsqueeze(-1).to(hidden.dtype)
        )
        return hidden


class MambaBranchEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        pad_id: int,
        max_length: int,
        hidden_dim: int,
        num_layers: int,
        ff_multiplier: int,
        dropout: float,
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> None:
        super().__init__()
        self.embedding = TokenPositionEmbedding(
            vocab_size=vocab_size,
            max_length=max_length,
            hidden_dim=hidden_dim,
            pad_id=pad_id,
            dropout=dropout,
        )
        block_cls: type[nn.Module]
        self.backend = "mamba_ssm" if MambaSSM is not None else "fallback_gated_conv"
        block_cls = OfficialMambaBlock if MambaSSM is not None else FallbackMambaBlock
        self.layers = nn.ModuleList(
            [
                block_cls(
                    hidden_dim=hidden_dim,
                    ff_multiplier=ff_multiplier,
                    dropout=dropout,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                if MambaSSM is not None
                else block_cls(
                    hidden_dim=hidden_dim,
                    ff_multiplier=ff_multiplier,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        hidden = self.embedding(tokens)
        for layer in self.layers:
            hidden = layer(hidden, mask)
        hidden = self.norm(hidden)
        return hidden * mask.unsqueeze(-1).to(hidden.dtype)


def build_mamba_model(
    *,
    protein_vocab_size: int,
    smiles_vocab_size: int,
    protein_pad_id: int,
    smiles_pad_id: int,
    protein_max_length: int,
    smiles_max_length: int,
    hidden_dim: int,
    num_layers: int,
    ff_multiplier: int,
    dropout: float,
    d_state: int,
    d_conv: int,
    expand: int,
) -> DualEncoderClassifier:
    protein_encoder = MambaBranchEncoder(
        vocab_size=protein_vocab_size,
        pad_id=protein_pad_id,
        max_length=protein_max_length,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        ff_multiplier=ff_multiplier,
        dropout=dropout,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    )
    smiles_encoder = MambaBranchEncoder(
        vocab_size=smiles_vocab_size,
        pad_id=smiles_pad_id,
        max_length=smiles_max_length,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        ff_multiplier=ff_multiplier,
        dropout=dropout,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    )
    classifier = FusionClassifier(hidden_dim=hidden_dim, dropout=dropout)
    return DualEncoderClassifier(
        protein_encoder=protein_encoder,
        smiles_encoder=smiles_encoder,
        classifier=classifier,
        model_name="mamba",
        mamba_backend=protein_encoder.backend,
    )
