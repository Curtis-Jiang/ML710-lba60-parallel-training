"""2-D hybrid parallel: tensor parallel within a TP group, manual DP across groups.

We deliberately avoid wrapping the TP-parallelized model in
``DistributedDataParallel``. DDP's initial ``_sync_module_states`` calls
``dist._broadcast_coalesced`` which dispatches ``aten.cat.default`` over the
module's full parameter tree -- including the DTensor-sharded FFN weights
produced by ``parallelize_module``. Torch 2.5's DTensor dispatcher refuses to
``cat`` a mix of plain ``Tensor`` and ``DTensor`` inputs, so DDP construction
crashes before a single step can run.

Instead we:

1. Build the 2-D device mesh ``(dp, tp)``.
2. Apply the TP plan along the ``tp`` dim (FFN linears become DTensors).
3. Attach the DP process group and size to the module. The trainer performs
   a single explicit per-parameter all-reduce across this group after
   ``loss.backward()`` (see ``_hybrid_dp_grad_allreduce`` in trainer.py).

Pre-condition assumed by the trainer: every rank initializes the model with
the same RNG seed (``config.experiment.seed``, no per-rank offset), so the
plain-tensor parameters are already identical across DP replicas and the
DTensor shards are consistent within each TP position. With that invariant
in place no initial broadcast is needed.
"""
from __future__ import annotations

from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from sequence_binding.engine.distributed import DistributedContext
from sequence_binding.engine.tensor_parallel import apply_tp_plan


def wrap_hybrid_tp_dp(
    model: nn.Module,
    config,
    context: DistributedContext,
    tp_wrap=None,  # unused; kept for API symmetry with trainer dispatch
) -> nn.Module:
    tp_size = int(config.distributed.tp_size)
    dp_size = int(config.distributed.dp_size)
    if tp_size * dp_size != context.world_size:
        raise ValueError(
            f"tp_size * dp_size ({tp_size}*{dp_size}) != world_size {context.world_size}"
        )
    mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
    apply_tp_plan(model, mesh["tp"])
    # Stash the DP group/size for the trainer's manual grad all-reduce hook.
    model._hybrid_dp_group = mesh["dp"].get_group()
    model._hybrid_dp_size = dp_size
    return model
