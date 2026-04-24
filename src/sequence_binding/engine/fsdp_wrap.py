"""FSDP wrapping for ZeRO-2 (SHARD_GRAD_OP) and ZeRO-3 (FULL_SHARD)."""
from __future__ import annotations

import functools

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from sequence_binding.engine.distributed import DistributedContext


_SHARD_MAP = {
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,  # ZeRO-2
    "full_shard": ShardingStrategy.FULL_SHARD,        # ZeRO-3
}


def _wrap_layer_classes() -> set[type[nn.Module]]:
    """Resolve transformer-layer classes that FSDP should treat as units."""
    layer_classes: set[type[nn.Module]] = {nn.TransformerEncoderLayer}
    try:
        from sequence_binding.models.mamba import MambaBlock

        layer_classes.add(MambaBlock)
    except Exception:  # pragma: no cover - mamba_ssm optional
        pass
    return layer_classes


def wrap_fsdp(model: nn.Module, config, context: DistributedContext) -> nn.Module:
    if not dist.is_initialized():
        raise RuntimeError("FSDP requires an initialized process group")
    strategy_name = config.distributed.fsdp_sharding
    if strategy_name not in _SHARD_MAP:
        raise ValueError(f"unknown fsdp_sharding {strategy_name!r}")

    if config.train.mixed_precision.lower() == "bf16":
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    else:
        mp = MixedPrecision()

    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=_wrap_layer_classes(),
    )
    return FSDP(
        model,
        sharding_strategy=_SHARD_MAP[strategy_name],
        auto_wrap_policy=auto_wrap,
        device_id=context.local_rank,
        mixed_precision=mp,
        use_orig_params=True,
    )
