from __future__ import annotations

from sequence_binding.config import ExperimentConfig
from sequence_binding.models.attention import build_attention_model
from sequence_binding.models.mamba import build_mamba_model


def build_model(config: ExperimentConfig, dataset_meta: dict):
    common = dict(
        protein_vocab_size=int(dataset_meta["protein_vocab_size"]),
        smiles_vocab_size=int(dataset_meta["smiles_vocab_size"]),
        protein_pad_id=int(dataset_meta["protein_pad_id"]),
        smiles_pad_id=int(dataset_meta["smiles_pad_id"]),
        protein_max_length=int(config.data.max_protein_len),
        smiles_max_length=int(config.data.max_smiles_len),
        hidden_dim=int(config.model.hidden_dim),
        num_layers=int(config.model.num_layers),
        ff_multiplier=int(config.model.ff_multiplier),
        dropout=float(config.model.dropout),
    )
    if config.model.name == "attention":
        return build_attention_model(
            **common,
            num_heads=int(config.model.num_heads),
        )
    if config.model.name == "mamba":
        return build_mamba_model(
            **common,
            d_state=int(config.model.mamba_d_state),
            d_conv=int(config.model.mamba_d_conv),
            expand=int(config.model.mamba_expand),
        )
    raise ValueError(f"unsupported model {config.model.name!r}")
