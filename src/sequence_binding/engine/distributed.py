from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    strategy: str
    device: torch.device
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        # Reflects reality rather than strategy name: branch_mp is a
        # "non-single" strategy in spirit but runs as one process with no
        # process group initialized, so collective helpers must treat it
        # as disabled. The only source of truth is ``dist.is_initialized``.
        return dist.is_available() and dist.is_initialized()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def branch_secondary_device() -> torch.device:
    """Secondary device used by the branch model-parallel wrapper."""
    return torch.device("cuda:1")


def setup_distributed(strategy: str, backend: str) -> DistributedContext:
    if strategy == "single":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return DistributedContext(
            strategy=strategy,
            device=device,
            rank=0,
            local_rank=0,
            world_size=1,
        )
    if strategy == "branch_mp":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("branch_mp requires >=2 CUDA devices in one process")
        primary = torch.device("cuda:0")
        torch.cuda.set_device(primary)
        return DistributedContext(
            strategy=strategy,
            device=primary,
            rank=0,
            local_rank=0,
            world_size=1,
        )
    if not torch.cuda.is_available():
        raise RuntimeError(f"{strategy} requires CUDA")
    # Keep NCCL defaults stable on the current cluster image.
    os.environ.setdefault("NCCL_NET_PLUGIN", "none")
    if "NCCL_TUNER_CONFIG_PATH" not in os.environ:
        for candidate in (
            "/usr/local/nvidia/lib64/a3plus_tuner_config.textproto",
            "/usr/local/nvidia/lib64/a3plus_tuner_config_ll128.textproto",
        ):
            if Path(candidate).exists():
                os.environ["NCCL_TUNER_CONFIG_PATH"] = candidate
                break
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)
    return DistributedContext(
        strategy=strategy,
        device=torch.device(f"cuda:{local_rank}"),
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.barrier()


def reduce_mean(value: float, context: DistributedContext) -> float:
    if not context.enabled:
        return float(value)
    tensor = torch.tensor([value], dtype=torch.float64, device=context.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= context.world_size
    return float(tensor.item())


def reduce_sum(value: float, context: DistributedContext) -> float:
    if not context.enabled:
        return float(value)
    tensor = torch.tensor([value], dtype=torch.float64, device=context.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def gather_numpy(array: np.ndarray, context: DistributedContext) -> list[np.ndarray]:
    if not context.enabled:
        return [array]
    gathered: list[np.ndarray | None] = [None for _ in range(context.world_size)]
    dist.all_gather_object(gathered, array)
    return [item for item in gathered if item is not None]
