"""Tensor parallel for the attention dual-branch encoder.

``nn.TransformerEncoderLayer`` stores its QKV weights in a single fused
``self_attn.in_proj_weight`` parameter whose DTensor layout is not directly
supported by ``parallelize_module``. We therefore shard the two FFN linears
(which are the bulk of the parameter count) with Megatron-style column/row
parallel and keep the attention replicated. This follows the FFN-only fallback
documented in magical-puzzling-spark.md section 13.
"""
from __future__ import annotations

from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from sequence_binding.engine.distributed import DistributedContext


def _tp_plan_for_encoder_layer() -> dict[str, object]:
    """Column-parallel expansion, row-parallel projection on the FFN."""
    return {
        "linear1": ColwiseParallel(),
        "linear2": RowwiseParallel(),
    }


def apply_tp_plan(model: nn.Module, tp_mesh) -> nn.Module:
    """Apply the TP plan to every TransformerEncoderLayer in both branches."""
    plan = _tp_plan_for_encoder_layer()
    for branch_name in ("protein_encoder", "smiles_encoder"):
        branch = getattr(model, branch_name, None)
        if branch is None or not hasattr(branch, "layers"):
            continue
        for layer in branch.layers:
            parallelize_module(layer, tp_mesh, plan)
    return model


def wrap_tp(model: nn.Module, config, context: DistributedContext) -> nn.Module:
    if config.model.name != "attention":
        raise ValueError("tp wrapper only supports the attention model")
    tp_size = int(config.distributed.tp_size)
    if tp_size < 2:
        raise ValueError(f"tp_size must be >= 2, got {tp_size}")
    mesh = init_device_mesh("cuda", (tp_size,), mesh_dim_names=("tp",))
    return apply_tp_plan(model, mesh["tp"])
