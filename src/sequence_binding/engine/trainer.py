from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sequence_binding.config import ExperimentConfig, resolve_repo_path
from sequence_binding.data import PairBindingTSVDataset, build_dataloader, load_char_tokenizer
from sequence_binding.engine.distributed import (
    DistributedContext,
    barrier,
    cleanup_distributed,
    gather_numpy,
    reduce_mean,
    reduce_sum,
    seed_everything,
    setup_distributed,
)
from sequence_binding.engine.metrics import binary_classification_metrics, finite_or_none
from sequence_binding.models import build_model


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _hybrid_dp_grad_allreduce(model: nn.Module) -> None:
    """Average gradients across the DP sub-mesh for hybrid_tp_dp.

    Replaces the role that ``DistributedDataParallel`` plays for pure-DDP:
    after ``loss.backward()`` every rank holds local gradients; we reduce
    them across the DP group so all DP replicas end up with the same
    averaged gradient. Works for both plain ``nn.Parameter`` (attention,
    classifier, embeddings, LayerNorm) and DTensor-sharded FFN parameters:
    for a DTensor grad we all-reduce the ``_local_tensor`` view, which is a
    plain ``cuda:local_rank`` tensor and therefore dispatches to ordinary
    NCCL all-reduce without tripping the DTensor mixed-type guard.
    """
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    dp_group = getattr(model, "_hybrid_dp_group", None)
    dp_size = int(getattr(model, "_hybrid_dp_size", 1))
    if dp_group is None or dp_size <= 1:
        return
    scale = 1.0 / dp_size
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad
        local = grad._local_tensor if isinstance(grad, DTensor) else grad
        dist.all_reduce(local, op=dist.ReduceOp.SUM, group=dp_group)
        local.mul_(scale)


def _clip_grad_norm(model: nn.Module, max_norm: float, strategy: str) -> None:
    """Grad-norm clipping that tolerates DTensor-sharded parameters.

    ``torch.nn.utils.clip_grad_norm_`` in Torch 2.5 defaults to the
    ``_foreach_norm`` kernel, which cannot consume a list containing both
    plain ``Tensor`` grads (replicated layers) and ``DTensor`` grads
    (columnwise/rowwise FFN shards). For TP / hybrid we therefore split the
    parameters by type and clip each group independently with
    ``foreach=False``. Every other strategy keeps the original fast path.
    """
    if strategy not in ("tp", "hybrid_tp_dp"):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return
    from torch.distributed.tensor import DTensor

    dtensor_params = [
        p for p in model.parameters() if p.requires_grad and isinstance(p, DTensor)
    ]
    plain_params = [
        p for p in model.parameters() if p.requires_grad and not isinstance(p, DTensor)
    ]
    if plain_params:
        torch.nn.utils.clip_grad_norm_(plain_params, max_norm, foreach=False)
    if dtensor_params:
        torch.nn.utils.clip_grad_norm_(dtensor_params, max_norm, foreach=False)


def _autocast_context(config: ExperimentConfig, device: torch.device):
    enabled = device.type == "cuda" and config.train.mixed_precision.lower() == "bf16"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def _ensure_batch_size(config: ExperimentConfig, data_world: int) -> int:
    if config.train.global_batch_size % data_world != 0:
        raise ValueError(
            f"global batch size {config.train.global_batch_size} must be divisible"
            f" by data-parallel world size {data_world}"
        )
    return config.train.global_batch_size // data_world


def _data_parallel_rank_world(
    strategy: str, config: ExperimentConfig, context: DistributedContext
) -> tuple[int, int]:
    """Rank/world used for data sharding.

    TP ranks within the same TP group must receive *identical* batches,
    otherwise the FFN all-reduce inside RowwiseParallel sees mismatched
    activation shapes and NCCL hangs. We therefore shard data only across
    the DP dimension of the mesh, not the TP dimension.
    """
    if strategy == "tp":
        return 0, 1
    if strategy == "hybrid_tp_dp":
        tp_size = int(config.distributed.tp_size)
        if tp_size < 1:
            raise ValueError(f"hybrid_tp_dp requires tp_size >= 1, got {tp_size}")
        return context.rank // tp_size, context.world_size // tp_size
    return context.rank, context.world_size


def _build_run_dir(config: ExperimentConfig, run_name: str | None) -> Path:
    base_dir = resolve_repo_path(config.experiment.output_dir)
    final_name = run_name or config.experiment.name
    run_dir = base_dir / final_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _build_optimizer(model: nn.Module, config: ExperimentConfig, strategy: str):
    params = [param for param in model.parameters() if param.requires_grad]
    if strategy == "ddp_zero":
        from torch.distributed.optim import ZeroRedundancyOptimizer

        return ZeroRedundancyOptimizer(
            params,
            optimizer_class=AdamW,
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
    # TP / hybrid_tp_dp: the parameter list is a mix of plain nn.Parameter
    # (attention QKV, embeddings, LayerNorm, classifier) and DTensor shards
    # (the FFN linears parallelized by ColwiseParallel/RowwiseParallel). Every
    # ``_foreach_*`` kernel in torch 2.5 refuses such a mixed list. The
    # single-tensor path dispatches per-parameter, letting each parameter's
    # __torch_dispatch__ handle its own kernel correctly.
    extra: dict[str, Any] = {}
    if strategy in ("tp", "hybrid_tp_dp"):
        extra = {"foreach": False, "fused": False}
    return AdamW(
        params,
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        **extra,
    )


def _forward_loss(
    model: nn.Module,
    batch,
    criterion: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
):
    batch = batch.to(device)
    with _autocast_context(config, device):
        logits = model(batch)
        loss = criterion(logits, batch.labels)
    return logits, loss, batch


def _load_dataset_meta(config: ExperimentConfig, protein_tokenizer, smiles_tokenizer) -> dict[str, int]:
    return {
        "protein_vocab_size": protein_tokenizer.vocab_size,
        "smiles_vocab_size": smiles_tokenizer.vocab_size,
        "protein_pad_id": protein_tokenizer.pad_id,
        "smiles_pad_id": smiles_tokenizer.pad_id,
    }


def _evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    config: ExperimentConfig,
    context: DistributedContext,
) -> dict[str, float]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if config.train.max_eval_batches is not None and batch_idx > config.train.max_eval_batches:
                break
            logits, loss, batch = _forward_loss(model, batch, criterion, config, context.device)
            labels_np = batch.labels.detach().cpu().numpy()
            logits_np = logits.detach().float().cpu().numpy()
            all_labels.append(labels_np)
            all_logits.append(logits_np)
            total_loss += float(loss.item()) * labels_np.shape[0]
            total_examples += int(labels_np.shape[0])
    gathered_logits = gather_numpy(np.concatenate(all_logits, axis=0), context)
    gathered_labels = gather_numpy(np.concatenate(all_labels, axis=0), context)
    total_loss = reduce_sum(total_loss, context)
    total_examples = int(reduce_sum(float(total_examples), context))
    if not context.is_main:
        return {}
    logits = np.concatenate(gathered_logits, axis=0)
    labels = np.concatenate(gathered_labels, axis=0)
    avg_loss = total_loss / max(total_examples, 1)
    return binary_classification_metrics(labels=labels, logits=logits, loss=avg_loss)


def run_training(
    config: ExperimentConfig,
    *,
    strategy: str | None = None,
    run_name: str | None = None,
) -> Path:
    strategy = strategy or config.distributed.strategy
    context = setup_distributed(strategy=strategy, backend=config.distributed.backend)
    run_dir = _build_run_dir(config, run_name)
    data_rank, data_world = _data_parallel_rank_world(strategy, config, context)
    # All ranks use the same seed so model construction is deterministic and
    # identical across the world. TP/hybrid require identical weights before
    # parallelize_module shards them; FSDP (sync_module_states=False in our
    # config) silently assumed the same. DDP would have broadcast rank-0's
    # state anyway, so a fixed seed is a strict improvement everywhere.
    # Per-rank diversity for sampling comes from DistributedSampler (which
    # seeds itself deterministically from epoch + seed).
    seed_everything(config.experiment.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    protein_tokenizer = load_char_tokenizer(config.data.protein_vocab_json)
    smiles_tokenizer = load_char_tokenizer(config.data.smiles_vocab_json)
    dataset_meta = _load_dataset_meta(config, protein_tokenizer, smiles_tokenizer)
    per_device_batch = _ensure_batch_size(config, data_world)
    train_dataset = PairBindingTSVDataset(
        tsv_path=config.data.train_tsv,
        protein_tokenizer=protein_tokenizer,
        smiles_tokenizer=smiles_tokenizer,
        max_protein_len=config.data.max_protein_len,
        max_smiles_len=config.data.max_smiles_len,
    )
    val_dataset = PairBindingTSVDataset(
        tsv_path=config.data.val_tsv,
        protein_tokenizer=protein_tokenizer,
        smiles_tokenizer=smiles_tokenizer,
        max_protein_len=config.data.max_protein_len,
        max_smiles_len=config.data.max_smiles_len,
    )
    # data_world == 1 => every rank iterates the full dataset in the same
    # order (enforced by DistributedSampler's deterministic shuffle + shared
    # seed above), which is the required invariant for pure TP.
    loader_distributed = context.enabled and data_world > 1
    train_loader, train_sampler = build_dataloader(
        train_dataset,
        batch_size=per_device_batch,
        protein_pad_id=protein_tokenizer.pad_id,
        smiles_pad_id=smiles_tokenizer.pad_id,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        distributed=loader_distributed,
        rank=data_rank,
        world_size=data_world,
        shuffle=True,
    )
    val_loader, _ = build_dataloader(
        val_dataset,
        batch_size=config.train.eval_batch_size,
        protein_pad_id=protein_tokenizer.pad_id,
        smiles_pad_id=smiles_tokenizer.pad_id,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        distributed=loader_distributed,
        rank=data_rank,
        world_size=data_world,
        shuffle=False,
    )
    model = build_model(config, dataset_meta).to(context.device)
    model_meta = _unwrap_model(model).metadata()
    if strategy in ("ddp", "ddp_zero"):
        model = DDP(
            model,
            device_ids=[context.local_rank],
            find_unused_parameters=config.distributed.find_unused_parameters,
        )
    elif strategy in ("fsdp_z2", "fsdp_z3"):
        from sequence_binding.engine.fsdp_wrap import wrap_fsdp

        model = wrap_fsdp(model, config, context)
    elif strategy == "branch_mp":
        from sequence_binding.engine.branch_parallel import wrap_branch_parallel

        model = wrap_branch_parallel(model, context)
    elif strategy == "tp":
        from sequence_binding.engine.tensor_parallel import wrap_tp

        model = wrap_tp(model, config, context)
    elif strategy == "hybrid_tp_dp":
        from sequence_binding.engine.hybrid import wrap_hybrid_tp_dp
        from sequence_binding.engine.tensor_parallel import wrap_tp

        model = wrap_hybrid_tp_dp(model, config, context, tp_wrap=wrap_tp)
    # strategy == "single": leave the model untouched.
    optimizer = _build_optimizer(model, config, strategy)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(len(train_loader) * config.train.epochs, 1))
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = float("-inf")
    best_metrics: dict[str, float] | None = None
    history: list[dict[str, Any]] = []
    best_checkpoint = run_dir / "best.pt"
    start_time = time.perf_counter()
    if context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)

    if context.is_main:
        for stale in ("history.jsonl", "summary.json", "best.pt"):
            stale_path = run_dir / stale
            if stale_path.exists():
                stale_path.unlink()
        _save_yaml(run_dir / "config_snapshot.yaml", config.to_dict())

    for epoch in range(config.train.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_start = time.perf_counter()
        train_loss_weighted = 0.0
        train_examples = 0
        for step, batch in enumerate(train_loader, start=1):
            if config.train.max_train_batches is not None and step > config.train.max_train_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            logits, loss, batch = _forward_loss(model, batch, criterion, config, context.device)
            loss.backward()
            if strategy == "hybrid_tp_dp":
                _hybrid_dp_grad_allreduce(model)
            _clip_grad_norm(model, config.train.grad_clip_norm, strategy)
            optimizer.step()
            scheduler.step()
            batch_examples = int(batch.labels.shape[0])
            train_loss_weighted += float(loss.item()) * batch_examples
            train_examples += batch_examples
            if context.is_main and step % config.train.log_interval == 0:
                print(
                    f"[train] epoch={epoch + 1} step={step}/{len(train_loader)} loss={loss.item():.4f}",
                    flush=True,
                )
        epoch_time = time.perf_counter() - epoch_start
        # Ranks that belong to the same TP group count the same batch, so the
        # global reduce double-counts by a factor of (world_size / data_world).
        tp_replication = max(context.world_size // max(data_world, 1), 1)
        raw_examples_sum = reduce_sum(float(train_examples), context)
        total_examples = int(raw_examples_sum / tp_replication)
        avg_train_loss = reduce_sum(train_loss_weighted, context) / max(raw_examples_sum, 1)
        train_examples_per_sec = total_examples / max(epoch_time, 1e-6)
        val_metrics = _evaluate(model, val_loader, criterion, config, context)
        barrier(context)
        if context.is_main:
            if context.device.type == "cuda":
                peak_gpu_bytes = int(torch.cuda.max_memory_allocated(context.device))
                torch.cuda.reset_peak_memory_stats(context.device)
            else:
                peak_gpu_bytes = 0
            epoch_payload = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_examples": total_examples,
                "train_examples_per_sec": train_examples_per_sec,
                "epoch_wall_time_sec": epoch_time,
                "cumulative_wall_time_sec": time.perf_counter() - start_time,
                "peak_gpu_bytes": peak_gpu_bytes,
                "val_metrics": val_metrics,
            }
            history.append(epoch_payload)
            with (run_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_payload) + "\n")
            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_metrics = val_metrics
                # FSDP / TP / hybrid model.state_dict() is a collective op that
                # all ranks must join; calling it only on rank 0 deadlocks. The
                # report pipeline reads summary.json + history, not best.pt, so
                # we skip the checkpoint for those strategies entirely.
                if strategy in ("fsdp_z2", "fsdp_z3", "tp", "hybrid_tp_dp"):
                    if epoch == 0:
                        print(
                            f"[info] skipping best.pt save for {strategy}"
                            " (sharded state_dict would deadlock across ranks)",
                            flush=True,
                        )
                else:
                    torch.save(
                        {
                            "model_state": _unwrap_model(model).state_dict(),
                            "model_metadata": asdict(model_meta),
                            "train_tsv": str(resolve_repo_path(config.data.train_tsv)),
                            "val_tsv": str(resolve_repo_path(config.data.val_tsv)),
                            "config": config.to_dict(),
                        },
                        best_checkpoint,
                    )
    total_wall_time_sec = time.perf_counter() - start_time
    if context.is_main:
        mean_throughput = float(np.mean([item["train_examples_per_sec"] for item in history]))
        best_epoch = max(
            history,
            key=lambda item: item["val_metrics"]["auc"],
        )["epoch"]
        summary = {
            "run_name": run_dir.name,
            "strategy": strategy,
            "model": config.model.name,
            "mamba_backend": model_meta.mamba_backend,
            "parameter_count": model_meta.parameter_count,
            "global_batch_size": config.train.global_batch_size,
            "per_device_batch_size": per_device_batch,
            "epochs": config.train.epochs,
            "train_tsv": str(resolve_repo_path(config.data.train_tsv)),
            "val_tsv": str(resolve_repo_path(config.data.val_tsv)),
            "best_epoch": int(best_epoch),
            "best_val_metrics": {
                key: finite_or_none(value) for key, value in (best_metrics or {}).items()
            },
            "mean_train_examples_per_sec": mean_throughput,
            "total_wall_time_sec": total_wall_time_sec,
            "peak_gpu_bytes_rank0": max(
                (item.get("peak_gpu_bytes", 0) for item in history), default=0
            ),
            "history": history,
            "history_path": str(run_dir / "history.jsonl"),
            "best_checkpoint": str(best_checkpoint),
        }
        _save_json(run_dir / "summary.json", summary)
    barrier(context)
    cleanup_distributed(context)
    return run_dir


def run_evaluation(
    config: ExperimentConfig,
    *,
    checkpoint_path: str | Path,
    split: str = "val",
    strategy: str = "single",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    context = setup_distributed(strategy=strategy, backend=config.distributed.backend)
    seed_everything(config.experiment.seed + context.rank)
    checkpoint = torch.load(
        resolve_repo_path(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    protein_tokenizer = load_char_tokenizer(config.data.protein_vocab_json)
    smiles_tokenizer = load_char_tokenizer(config.data.smiles_vocab_json)
    dataset_meta = _load_dataset_meta(config, protein_tokenizer, smiles_tokenizer)
    dataset = PairBindingTSVDataset(
        tsv_path=(config.data.train_tsv if split == "train" else config.data.val_tsv),
        protein_tokenizer=protein_tokenizer,
        smiles_tokenizer=smiles_tokenizer,
        max_protein_len=config.data.max_protein_len,
        max_smiles_len=config.data.max_smiles_len,
    )
    loader, _ = build_dataloader(
        dataset,
        batch_size=config.train.eval_batch_size,
        protein_pad_id=protein_tokenizer.pad_id,
        smiles_pad_id=smiles_tokenizer.pad_id,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        distributed=context.enabled,
        rank=context.rank,
        world_size=context.world_size,
        shuffle=False,
    )
    model = build_model(config, dataset_meta).to(context.device)
    model.load_state_dict(checkpoint["model_state"])
    if context.enabled:
        model = DDP(
            model,
            device_ids=[context.local_rank],
            find_unused_parameters=config.distributed.find_unused_parameters,
        )
    criterion = nn.BCEWithLogitsLoss()
    metrics = _evaluate(model, loader, criterion, config, context)
    if context.is_main and output_path is not None:
        _save_json(resolve_repo_path(output_path), metrics)
    barrier(context)
    cleanup_distributed(context)
    return metrics if context.is_main else {}
