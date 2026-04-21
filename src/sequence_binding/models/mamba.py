from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch import Tensor, nn

from sequence_binding.models.common import (
    DualEncoderClassifier,
    FusionClassifier,
    TokenPositionEmbedding,
)

def _load_mamba_cls():
    try:
        from mamba_ssm import Mamba as MambaSSM  # type: ignore

        return MambaSSM
    except Exception:  # pragma: no cover - optional dependency
        pass

    # Some environments can install `mamba-ssm` successfully but fail on the
    # top-level import because that path pulls in optional generation utilities.
    # For this project we only need the core sequence block, so we load the
    # `modules/mamba_simple.py` implementation directly when available.
    try:
        for key in list(sys.modules):
            if key == "mamba_ssm" or key.startswith("mamba_ssm."):
                sys.modules.pop(key, None)

        spec = importlib.util.find_spec("mamba_ssm")
        if spec is None or not spec.submodule_search_locations:
            return None
        package_dir = Path(next(iter(spec.submodule_search_locations)))
        package = types.ModuleType("mamba_ssm")
        package.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
        package.__package__ = "mamba_ssm"
        sys.modules["mamba_ssm"] = package

        module_name = "mamba_ssm.modules.mamba_simple"
        module_path = package_dir / "modules" / "mamba_simple.py"
        module_spec = importlib.util.spec_from_file_location(module_name, module_path)
        if module_spec is None or module_spec.loader is None:
            return None
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
        return getattr(module, "Mamba", None)
    except Exception:  # pragma: no cover - optional dependency
        return None


MambaSSM = _load_mamba_cls()


class MambaBlock(nn.Module):
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
            raise RuntimeError(
                "mamba-ssm is required for the final project version. "
                "Run `bash scripts/install_mamba.sh` first."
            )
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
        self.backend = "mamba_ssm"
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    hidden_dim=hidden_dim,
                    ff_multiplier=ff_multiplier,
                    dropout=dropout,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
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
