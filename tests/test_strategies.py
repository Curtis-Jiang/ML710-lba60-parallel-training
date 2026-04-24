"""Unit tests for the new parallel strategies (CPU-friendly portions only)."""
from __future__ import annotations

import pytest
import torch

from sequence_binding.engine.metrics import goodput


def test_goodput_formula() -> None:
    # Perfect convergence and perfect scaling -> goodput == throughput.
    assert goodput(
        examples_per_sec=1000.0,
        val_auc=0.9,
        baseline_val_auc=0.9,
        scaling_efficiency=1.0,
    ) == pytest.approx(1000.0)
    # Half scaling efficiency halves goodput.
    assert goodput(
        examples_per_sec=1000.0,
        val_auc=0.9,
        baseline_val_auc=0.9,
        scaling_efficiency=0.5,
    ) == pytest.approx(500.0)
    # Negative scaling efficiency is clamped at zero.
    assert goodput(
        examples_per_sec=1000.0,
        val_auc=0.9,
        baseline_val_auc=0.9,
        scaling_efficiency=-0.3,
    ) == 0.0


def test_tp_plan_dict_structure() -> None:
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    from sequence_binding.engine.tensor_parallel import _tp_plan_for_encoder_layer

    plan = _tp_plan_for_encoder_layer()
    assert set(plan.keys()) == {"linear1", "linear2"}
    assert isinstance(plan["linear1"], ColwiseParallel)
    assert isinstance(plan["linear2"], RowwiseParallel)


def test_hybrid_world_size_assert() -> None:
    from types import SimpleNamespace

    from sequence_binding.engine.distributed import DistributedContext
    from sequence_binding.engine.hybrid import wrap_hybrid_tp_dp

    config = SimpleNamespace(
        distributed=SimpleNamespace(
            tp_size=2, dp_size=2, find_unused_parameters=False
        )
    )
    ctx = DistributedContext(
        strategy="hybrid_tp_dp",
        device=torch.device("cpu"),
        rank=0,
        local_rank=0,
        world_size=3,  # intentionally mismatched
    )
    with pytest.raises(ValueError, match="world_size"):
        wrap_hybrid_tp_dp(torch.nn.Linear(4, 4), config, ctx)


def test_fsdp_wrap_importable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("fsdp_wrap requires CUDA to exercise wrap_fsdp")
    from sequence_binding.engine import fsdp_wrap

    assert hasattr(fsdp_wrap, "wrap_fsdp")
    assert callable(fsdp_wrap.wrap_fsdp)


def test_branch_parallel_forward_two_devices() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("branch_mp requires >=2 CUDA devices")
    from sequence_binding.data.batch import PairBatch
    from sequence_binding.engine.branch_parallel import BranchParallelWrapper
    from sequence_binding.models.attention import build_attention_model

    primary = torch.device("cuda:0")
    secondary = torch.device("cuda:1")
    model = build_attention_model(
        protein_vocab_size=32,
        smiles_vocab_size=32,
        protein_pad_id=0,
        smiles_pad_id=0,
        protein_max_length=8,
        smiles_max_length=8,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        ff_multiplier=2,
        dropout=0.0,
    )
    wrapped = BranchParallelWrapper(model, primary=primary, secondary=secondary)
    batch = PairBatch(
        protein_tokens=torch.tensor([[1, 2, 3]], dtype=torch.long),
        protein_mask=torch.tensor([[1, 1, 1]], dtype=torch.bool),
        smiles_tokens=torch.tensor([[4, 5, 0]], dtype=torch.long),
        smiles_mask=torch.tensor([[1, 1, 0]], dtype=torch.bool),
        labels=torch.tensor([1.0], dtype=torch.float32),
    ).to(primary)
    logits = wrapped(batch)
    assert logits.device == primary
    assert tuple(logits.shape) == (1,)
