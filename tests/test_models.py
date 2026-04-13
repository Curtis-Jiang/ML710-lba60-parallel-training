from __future__ import annotations

import torch

from sequence_binding.data.batch import PairBatch
from sequence_binding.models.attention import build_attention_model
from sequence_binding.models.mamba import build_mamba_model


def _dummy_batch() -> PairBatch:
    return PairBatch(
        protein_tokens=torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long),
        protein_mask=torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        smiles_tokens=torch.tensor([[6, 7, 0], [8, 9, 10]], dtype=torch.long),
        smiles_mask=torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        labels=torch.tensor([0.0, 1.0], dtype=torch.float32),
    )


def test_attention_forward_backward() -> None:
    model = build_attention_model(
        protein_vocab_size=32,
        smiles_vocab_size=32,
        protein_pad_id=0,
        smiles_pad_id=0,
        protein_max_length=8,
        smiles_max_length=8,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        ff_multiplier=4,
        dropout=0.1,
    )
    batch = _dummy_batch()
    logits = model(batch)
    assert tuple(logits.shape) == (2,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch.labels)
    loss.backward()


def test_mamba_forward_backward() -> None:
    model = build_mamba_model(
        protein_vocab_size=32,
        smiles_vocab_size=32,
        protein_pad_id=0,
        smiles_pad_id=0,
        protein_max_length=8,
        smiles_max_length=8,
        hidden_dim=32,
        num_layers=2,
        ff_multiplier=4,
        dropout=0.1,
        d_state=8,
        d_conv=4,
        expand=2,
    )
    batch = _dummy_batch()
    logits = model(batch)
    assert tuple(logits.shape) == (2,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch.labels)
    loss.backward()
