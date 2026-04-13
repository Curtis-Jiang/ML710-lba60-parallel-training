from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

from _path_setup import ensure_cuda_libs_visible, setup_path

ensure_cuda_libs_visible()
setup_path()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.utils.data import DataLoader, DistributedSampler, Subset  # noqa: E402

from binding_affinity.data.collate import collate_lba_samples  # noqa: E402
from binding_affinity.data.processed_dataset import ProcessedLBADataset  # noqa: E402
from binding_affinity.losses.balanced_mse import BalancedMSEConfig, BalancedMSELoss, make_histogram_bins  # noqa: E402
from binding_affinity.losses.ranking import (  # noqa: E402
    ListwiseNDCGConfig,
    PairwiseRankingConfig,
    listwise_ndcg_loss,
    pairwise_ranking_loss,
)
from binding_affinity.models.affinity.encoder_faenet import FAENetConfig  # noqa: E402
from binding_affinity.models.affinity.energy_head import EnergyHeadConfig  # noqa: E402
from binding_affinity.models.affinity.model import AffinityModel, AffinityModelConfig  # noqa: E402
from binding_affinity.models.affinity.regressor import RegressorConfig  # noqa: E402
from binding_affinity.models.affinity.tokenizer_hier import TokenizerConfig  # noqa: E402
from binding_affinity.utils.ckpt import load_checkpoint, save_checkpoint  # noqa: E402
from binding_affinity.utils.config import load_config, set_by_dotted_key  # noqa: E402
from binding_affinity.utils.logger import JsonlLogger  # noqa: E402
from binding_affinity.utils.metrics import apply_affine, compute_metrics, fit_affine_calibration  # noqa: E402
from binding_affinity.utils.paths import ensure_dir, resolve_ws_path  # noqa: E402
from binding_affinity.utils.seed import configure_determinism, set_seed  # noqa: E402


def _init_distributed() -> Tuple[bool, int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # This container ships NCCL plugins under `/usr/local/nvidia/lib64` that require
    # extra env config; prefer the internal network plugin for stability.
    os.environ.setdefault("NCCL_NET_PLUGIN", "none")
    # NCCL tuner plugin crashes if this is missing in this environment.
    if "NCCL_TUNER_CONFIG_PATH" not in os.environ:
        for cand in [
            "/usr/local/nvidia/lib64/a3plus_tuner_config.textproto",
            "/usr/local/nvidia/lib64/a3plus_tuner_config_ll128.textproto",
        ]:
            if Path(cand).exists():
                os.environ["NCCL_TUNER_CONFIG_PATH"] = cand
                break
    # IMPORTANT: set device before NCCL init; otherwise all ranks may initialize on device 0.
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, rank, world, local_rank


def _is_rank0(rank: int) -> bool:
    return int(rank) == 0


def _build_model(cfg: Dict[str, Any], *, state_dict: Optional[Dict[str, Any]] = None) -> AffinityModel:
    mc = cfg["model"]
    d_model = int(mc["d_model"])
    z_embedding = mc.get("z_embedding", None)
    z_vocab = mc.get("z_vocab", None)
    if z_embedding is None and state_dict is not None:
        if any(k.startswith("encoder.elem_emb.") for k in state_dict.keys()):
            z_embedding = "factorized"
        elif "encoder.z_emb.weight" in state_dict:
            z_embedding = "combined"
            z_vocab = int(state_dict["encoder.z_emb.weight"].shape[0])
    if z_embedding is None:
        z_embedding = "combined"
    enc = FAENetConfig(
        d_model=d_model,
        layers=int(mc["layers"]),
        rbf_kernels=int(mc["rbf_kernels"]),
        cutoff=float(mc["cutoff"]),
        intra_cutoff=mc.get("intra_cutoff", None),
        use_ligand_flag=bool(mc.get("use_ligand_flag", True)),
        z_embedding=str(z_embedding),
        z_vocab=int(z_vocab) if z_vocab is not None else 128,
        edge_swish=bool(mc.get("edge_swish", False)),
        use_bond_emb=bool(mc.get("use_bond_emb", False)),
        graph_norm=bool(mc.get("graph_norm", False)),
        graph_norm_eps=float(mc.get("graph_norm_eps", 1e-5)),
        edge_type_mode=str(mc.get("edge_type_mode", "shared")),
        interaction_mode=str(mc.get("interaction_mode", "single_stream")),
        envelope=str(mc.get("envelope", "none")),
        use_atom_name_emb=bool(mc.get("use_atom_name_emb", False)),
        atom_name_vocab=int(mc.get("atom_name_vocab", 64)),
        ligand_feat_dim=int(mc.get("ligand_feat_dim", 0)),
    )
    eh = EnergyHeadConfig(d_model=d_model, hidden=int(mc["energy_head"]["hidden"]))

    tok_cfg = None
    tok = mc.get("tokenizer") or {}
    if bool(tok.get("enabled", True)):
        tok_cfg = TokenizerConfig(
            d_model=d_model,
            kp=int(tok["kp"]),
            kl=int(tok["kl"]),
            kint=int(tok["kint"]),
            attn_heads=int(tok.get("attn_heads", 4)),
            dropout=float(tok.get("dropout", 0.0)),
        )

    reg = RegressorConfig(
        d_model=d_model,
        hidden=int(mc["regressor"]["hidden"]),
        dropout=float(mc["regressor"].get("dropout", 0.0)),
        cross_enabled=bool(mc["regressor"].get("cross_enabled", False)),
    )
    return AffinityModel(AffinityModelConfig(encoder=enc, energy_head=eh, tokenizer=tok_cfg, regressor=reg))


def _dataset_labels_for_balanced_mse(ds: Any) -> Optional[List[float]]:
    # Fast path for ProcessedLBADataset.
    samples = getattr(ds, "samples", None)
    if isinstance(samples, list) and samples and isinstance(samples[0], dict) and ("y" in samples[0]):
        return [float(s["y"]) for s in samples]

    # Some alternate datasets store labels in a list under `_y`.
    ys = getattr(ds, "_y", None)
    if isinstance(ys, list) and ys:
        return [float(y) for y in ys]

    if isinstance(ds, Subset):
        base = _dataset_labels_for_balanced_mse(ds.dataset)
        if base is None:
            return None
        return [float(base[int(i)]) for i in ds.indices]
    return None


def _safe_clip_grad_norm_fp64_(
    parameters,
    max_norm: float,
    *,
    eps: float = 1e-12,
    fp32_max_abs_threshold: float = 1e19,
) -> torch.Tensor:
    """Clip grads like torch.nn.utils.clip_grad_norm_, but avoid fp32 overflow when grads are huge.

    Fast-path: if max(|grad|) is below a safe threshold, defer to torch's native fp32 implementation
    to preserve historical behavior.
    """
    params = [p for p in parameters if (p is not None and getattr(p, "grad", None) is not None)]
    if not params:
        return torch.zeros((), dtype=torch.float32)

    device = params[0].grad.device  # type: ignore[union-attr]
    max_abs = 0.0
    for p in params:
        g = p.grad.detach()  # type: ignore[union-attr]
        if g.is_sparse:
            g = g.coalesce().values()
        if not torch.isfinite(g).all():
            return torch.tensor(float("nan"), device=device, dtype=torch.float32)
        if g.numel():
            max_abs = max(max_abs, float(g.detach().float().abs().max().item()))

    if max_abs < float(fp32_max_abs_threshold):
        return torch.nn.utils.clip_grad_norm_(params, max_norm)

    total_sq = torch.zeros((), device=device, dtype=torch.float64)
    for p in params:
        g = p.grad.detach()  # type: ignore[union-attr]
        if g.is_sparse:
            g = g.coalesce().values()
        gd = g.double()
        total_sq = total_sq + (gd * gd).sum()

    total_norm = torch.sqrt(total_sq)
    if not torch.isfinite(total_norm):
        return total_norm.to(dtype=torch.float32)

    coef = float(max_norm) / (total_norm + float(eps))
    if coef < 1.0:
        coef_t = torch.as_tensor(coef, device=device, dtype=torch.float32)
        for p in params:
            p.grad.detach().mul_(coef_t)  # type: ignore[union-attr]
    return total_norm.to(dtype=torch.float32)


def _load_teacher_maps_from_parquet(
    parquet_path: str,
    *,
    pred_col: str,
    split: str = "train",
    weight_col: Optional[str] = None,
) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
    abs_path = str(resolve_ws_path(parquet_path))
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Distillation requires `pandas` to read parquet teacher preds.") from e

    cols = ["id", pred_col]
    if weight_col is not None:
        cols.append(str(weight_col))
    if split is not None:
        cols.append("split")
    df = pd.read_parquet(abs_path, columns=cols)
    if pred_col not in df.columns:
        raise ValueError(f"Teacher parquet missing column {pred_col!r}: {abs_path}")
    if weight_col is not None and str(weight_col) not in df.columns:
        raise ValueError(f"Teacher parquet missing column {str(weight_col)!r}: {abs_path}")
    if split is not None:
        if "split" not in df.columns:
            raise ValueError(f"Teacher parquet missing column 'split' (needed for split={split!r}): {abs_path}")
        df = df[df["split"] == split]

    ids = df["id"].astype(str).tolist()
    preds = df[pred_col].astype(float).tolist()
    pred_map = {str(i): float(p) for i, p in zip(ids, preds)}
    if not pred_map:
        raise ValueError(f"Empty teacher map from parquet (split={split!r}): {abs_path}")
    weight_map = None
    if weight_col is not None:
        ws = df[str(weight_col)].astype(float).tolist()
        weight_map = {str(i): float(w) for i, w in zip(ids, ws)}
    return pred_map, weight_map


@torch.no_grad()
def _eval_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    trues = []
    for batch in loader:
        for k in ["complex", "protein", "ligand"]:
            batch[k] = batch[k].to(device)
        y = batch["y"].to(device)
        pred, _ = model(batch)
        preds.append(pred.detach().float().cpu())
        trues.append(y.detach().float().cpu())
    pred_all = torch.cat(preds).numpy()
    true_all = torch.cat(trues).numpy()
    return pred_all, true_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the binding-affinity model on processed LBA tensors.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--init_from", type=str, default=None, help="Path to checkpoint to initialize weights from (finetune).")
    parser.add_argument(
        "--init_from_scope",
        type=str,
        default="full",
        choices=["full", "encoder", "encoder_energy"],
        help="Which parts to load from --init_from (default: full).",
    )
    parser.add_argument("--set", action="append", default=[], help="Override config: key=value (dotted keys supported).")
    args = parser.parse_args()

    if args.resume and args.init_from:
        raise ValueError("--resume and --init_from are mutually exclusive")

    distributed, rank, world, local_rank = _init_distributed()
    is_rank0 = _is_rank0(rank)

    cfg = load_config(args.config)
    for kv in args.set:
        if "=" not in kv:
            raise ValueError(f"--set expects key=value, got {kv}")
        k, v = kv.split("=", 1)
        set_by_dotted_key(cfg, k.strip(), yaml.safe_load(v))

    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    seed = int(cfg.get("seed", 0))
    set_seed(seed + rank)
    configure_determinism()

    task = cfg["task"]["name"]
    io = cfg["io"]
    processed_root = resolve_ws_path(io["processed_root"])
    runs_root = ensure_dir(io["runs_root"])

    run_dir = Path(runs_root) / task
    if args.run_name:
        run_dir = run_dir / str(args.run_name)
    run_dir = ensure_dir(run_dir)
    if is_rank0:
        (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_rank0:
        print(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "device": str(device),
                "distributed": distributed,
                "world_size": world,
            }
        )

    train_cfg = cfg["train"]
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 0))
    epochs = int(train_cfg["epochs"])

    data_cfg = cfg.get("data") or {}
    data_kind = str(data_cfg.get("kind", "processed_lba")).lower()
    if data_kind in {"processed_lba", "atom3d_processed", "processed"}:
        train_ds: Any = ProcessedLBADataset(processed_root / f"{task}_train.pt")
        val_ds: Any = ProcessedLBADataset(processed_root / f"{task}_val.pt")
        test_ds: Any = ProcessedLBADataset(processed_root / f"{task}_test.pt")
    else:
        raise ValueError(
            f"Unsupported data.kind: {data_kind}. This standalone ML710 project only supports preprocessed LBA tensors."
        )

    # Optional debug limits (keep deterministic).
    lt = train_cfg.get("limit_train_samples")
    lv = train_cfg.get("limit_val_samples")
    lte = train_cfg.get("limit_test_samples")
    if lt is not None:
        train_ds = Subset(train_ds, range(int(lt)))
    if lv is not None:
        val_ds = Subset(val_ds, range(int(lv)))
    if lte is not None:
        test_ds = Subset(test_ds, range(int(lte)))

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_lba_samples,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_lba_samples)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_lba_samples)

    resume_obj = None
    resume_state = None
    if args.resume:
        resume_obj = load_checkpoint(args.resume, map_location="cpu")
        resume_state = resume_obj["model"]

    init_obj = None
    init_state = None
    if args.init_from:
        init_obj = load_checkpoint(args.init_from, map_location="cpu")
        init_state = init_obj["model"]

    model = _build_model(cfg, state_dict=(resume_state or init_state)).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Optional EMA (rank0 only; model params are synced under DDP).
    ema_cfg = (train_cfg.get("ema") or {}) if isinstance(train_cfg, dict) else {}
    ema_enabled = bool(ema_cfg.get("enabled", False))
    ema_decay = float(ema_cfg.get("decay", 0.999))
    ema_model = None
    if ema_enabled and is_rank0:
        base = model.module if isinstance(model, DDP) else model
        ema_model = copy.deepcopy(base).to(device)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        ema_model.eval()

    # Optional distillation: train against a teacher prediction aligned by sample id.
    distill_cfg = train_cfg.get("distill") if isinstance(train_cfg, dict) else None
    distill_cfg = distill_cfg if isinstance(distill_cfg, dict) else {}
    distill_enabled = bool(distill_cfg.get("enabled", False))
    distill_alpha_start = float(distill_cfg.get("alpha", distill_cfg.get("weight", 0.5)))
    distill_alpha_end = float(distill_cfg.get("alpha_end", distill_alpha_start))
    teacher_map: Optional[Dict[str, float]] = None
    teacher_weight_map: Optional[Dict[str, float]] = None
    teacher_weight_col = None
    teacher_weight_clip_max = float(distill_cfg.get("teacher_weight_clip_max", 10.0))
    if distill_enabled:
        if not (0.0 <= distill_alpha_start <= 1.0):
            raise ValueError(f"train.distill.alpha must be in [0,1], got {distill_alpha_start}")
        if not (0.0 <= distill_alpha_end <= 1.0):
            raise ValueError(f"train.distill.alpha_end must be in [0,1], got {distill_alpha_end}")
        teacher_parquet = distill_cfg.get("teacher_parquet", None)
        if teacher_parquet is None:
            raise ValueError("train.distill.teacher_parquet is required when distillation is enabled.")
        teacher_pred_col = str(distill_cfg.get("teacher_pred_col", "pred_mix"))
        teacher_split = str(distill_cfg.get("teacher_split", "train"))
        teacher_weight_col = distill_cfg.get("teacher_weight_col", None)
        if teacher_weight_col is not None:
            teacher_weight_col = str(teacher_weight_col)
        teacher_map, teacher_weight_map = _load_teacher_maps_from_parquet(
            str(teacher_parquet),
            pred_col=teacher_pred_col,
            split=teacher_split,
            weight_col=teacher_weight_col,
        )

    # Losses
    loss_cfg = cfg["loss"]
    loss_name = str(loss_cfg.get("name", "balanced_mse"))
    if loss_name == "balanced_mse":
        bm_cfg = BalancedMSEConfig(
            num_bins=int(loss_cfg["balanced_mse"]["num_bins"]),
            init_sigma=float(loss_cfg["balanced_mse"]["init_sigma"]),
        )
        labels = _dataset_labels_for_balanced_mse(train_ds)
        if labels is None:
            # Fallback: this may be slow for datasets that load graphs in __getitem__.
            labels = [float(train_ds[i]["y"]) for i in range(len(train_ds))]
        y_train = torch.tensor(labels, dtype=torch.float32)
        centers, probs = make_histogram_bins(y_train, num_bins=bm_cfg.num_bins)
        reg_loss = BalancedMSELoss(bin_centers=centers, bin_probs=probs, cfg=bm_cfg).to(device)
    elif loss_name == "mse":
        reg_loss = torch.nn.MSELoss().to(device)
    else:
        raise ValueError(f"Unknown regression loss: {loss_name}")

    rank_section = loss_cfg.get("ranking") or {}
    rank_name = str(rank_section.get("name", "pairwise_softplus"))
    w_rank = float(loss_cfg["weights"].get("rank", 0.0)) if bool(rank_section.get("enabled", True)) else 0.0
    w_loc = float(loss_cfg["weights"].get("loc", 0.0))
    w_ent = float(loss_cfg["weights"].get("entropy", 0.0))

    rank_cfg_pair = PairwiseRankingConfig(margin=float(rank_section.get("margin", 0.0)))
    rank_cfg_ndcg = ListwiseNDCGConfig(
        temperature=float(rank_section.get("temperature", 1.0)),
        exp_gain_scale=float(rank_section.get("exp_gain_scale", 1.0)),
    )

    # Optim / sched
    optim_cfg = cfg["optim"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(optim_cfg["lr"]), weight_decay=float(optim_cfg.get("weight_decay", 0.0)))
    sched_cfg = cfg["sched"]
    sched_name = str(sched_cfg.get("name", "warmup_cosine"))
    warmup_epochs = int(sched_cfg.get("warmup_epochs", 0))
    min_lr = float(sched_cfg.get("min_lr", 0.0))

    lr_max = float(optim_cfg["lr"])

    def lr_at_epoch(epoch_idx: int) -> float:
        if epoch_idx < warmup_epochs:
            return lr_max * float(epoch_idx + 1) / max(warmup_epochs, 1)
        if sched_name in {"constant", "warmup_constant"}:
            return lr_max
        if sched_name in {"warmup_cosine", "cosine"}:
            t = float(epoch_idx - warmup_epochs) / max(epochs - warmup_epochs, 1)
            cos = 0.5 * (1.0 + math.cos(math.pi * t))
            return min_lr + (lr_max - min_lr) * cos
        raise ValueError(f"Unknown sched.name: {sched_name}")

    # Resume
    start_epoch = 0
    best_metric: Optional[float] = None
    best_path = run_dir / "ckpt_best.pt"
    if args.resume:
        ckpt = resume_obj if resume_obj is not None else load_checkpoint(args.resume, map_location="cpu")
        state = ckpt.get("model_raw", None)
        (model.module if isinstance(model, DDP) else model).load_state_dict(ckpt["model"] if state is None else state, strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metric = ckpt.get("best_metric", None)
    elif args.init_from:
        ckpt = init_obj if init_obj is not None else load_checkpoint(args.init_from, map_location="cpu")
        init_state_dict = ckpt.get("model_raw", None) or ckpt["model"]
        init_scope = str(args.init_from_scope or "full").lower()
        if init_scope == "full":
            (model.module if isinstance(model, DDP) else model).load_state_dict(init_state_dict, strict=True)
        else:
            prefixes = ("encoder.",) if init_scope == "encoder" else ("encoder.", "energy_head.")
            filtered = {k: v for k, v in init_state_dict.items() if any(k.startswith(p) for p in prefixes)}
            missing, unexpected = (model.module if isinstance(model, DDP) else model).load_state_dict(filtered, strict=False)
            if is_rank0:
                print(
                    {
                        "init_from": str(args.init_from),
                        "init_from_scope": init_scope,
                        "loaded_keys": int(len(filtered)),
                        "missing_keys": int(len(missing)),
                        "unexpected_keys": int(len(unexpected)),
                    }
                )

    logger = JsonlLogger(run_dir / "metrics.jsonl") if is_rank0 else None

    save_best_key = str(train_cfg.get("save_best_metric", "val/pearson"))
    best_mode = "max" if save_best_key.endswith("pearson") or save_best_key.endswith("spearman") else "min"
    grad_clip_norm_raw = train_cfg.get("grad_clip_norm", 1.0)
    grad_clip_norm = None if grad_clip_norm_raw is None else float(grad_clip_norm_raw)
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        grad_clip_norm = None

    for epoch in range(start_epoch, epochs):
        lr = lr_at_epoch(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        distill_alpha_epoch = 0.0
        if distill_enabled:
            if epochs <= 1:
                distill_alpha_epoch = float(distill_alpha_start)
            else:
                t = float(epoch) / float(max(epochs - 1, 1))
                distill_alpha_epoch = float(distill_alpha_start) + (float(distill_alpha_end) - float(distill_alpha_start)) * t

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        loss_sum = 0.0
        n_sum = 0
        max_grad_norm = 0.0
        nan_inf_steps = 0
        num_steps = 0
        num_updates = 0
        t0 = time.time()
        t_train0 = time.time()

        for step, batch in enumerate(train_loader):
            num_steps += 1
            for k in ["complex", "protein", "ligand"]:
                batch[k] = batch[k].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                pred, debug = model(batch)
                loss_reg_true = reg_loss(pred, y)
                if distill_enabled and teacher_map is not None and distill_alpha_epoch > 0.0:
                    ids = batch.get("id", None)
                    if not isinstance(ids, list):
                        raise RuntimeError("Distillation expects batch['id'] as a list[str].")
                    missing = [str(i) for i in ids if str(i) not in teacher_map]
                    if missing:
                        raise KeyError(f"Missing {len(missing)} ids in teacher_map (first={missing[0]!r}).")
                    y_teacher = torch.tensor([teacher_map[str(i)] for i in ids], device=device, dtype=y.dtype)
                    diff = pred - y_teacher
                    if teacher_weight_col is not None and teacher_weight_map is not None:
                        missing_w = [str(i) for i in ids if str(i) not in teacher_weight_map]
                        if missing_w:
                            raise KeyError(f"Missing {len(missing_w)} ids in teacher_weight_map (first={missing_w[0]!r}).")
                        w = torch.tensor([teacher_weight_map[str(i)] for i in ids], device=device, dtype=y.dtype)
                        if str(teacher_weight_col).startswith("logv_"):
                            w = torch.exp(-w)
                        if teacher_weight_clip_max is not None and teacher_weight_clip_max > 0:
                            w = w.clamp(max=float(teacher_weight_clip_max))
                        w = w / w.mean().clamp_min(1e-8)
                        loss_reg_teacher = (w * (diff * diff)).mean()
                    else:
                        loss_reg_teacher = (diff * diff).mean()
                    loss_reg = (1.0 - distill_alpha_epoch) * loss_reg_true + distill_alpha_epoch * loss_reg_teacher
                else:
                    loss_reg = loss_reg_true
                if w_rank > 0:
                    if rank_name == "pairwise_softplus":
                        loss_rank = pairwise_ranking_loss(pred, y, cfg=rank_cfg_pair)
                    elif rank_name == "listwise_ndcg":
                        loss_rank = listwise_ndcg_loss(pred, y, cfg=rank_cfg_ndcg)
                    else:
                        raise ValueError(f"Unknown ranking loss: {rank_name}")
                else:
                    loss_rank = torch.zeros((), device=device)
                loss_loc = debug.get("loc", torch.zeros((), device=device))
                loss_ent = debug.get("entropy", torch.zeros((), device=device))
                loss = loss_reg + w_rank * loss_rank + w_loc * loss_loc + w_ent * loss_ent
            if not torch.isfinite(loss).all():
                nan_inf_steps += 1
                continue
            loss.backward()
            gn = _safe_clip_grad_norm_fp64_(model.parameters(), grad_clip_norm if grad_clip_norm is not None else 1e9)
            if not torch.isfinite(gn).all():
                nan_inf_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            max_grad_norm = max(max_grad_norm, float(gn.detach().cpu()))
            optimizer.step()
            num_updates += 1
            if ema_enabled and ema_model is not None:
                with torch.no_grad():
                    base = model.module if isinstance(model, DDP) else model
                    for ema_p, p in zip(ema_model.parameters(), base.parameters()):
                        ema_p.mul_(ema_decay).add_(p.detach(), alpha=(1.0 - ema_decay))

            bs = int(y.shape[0])
            loss_sum += float(loss.detach().cpu()) * bs
            n_sum += bs

        # Reduce train loss across ranks.
        if distributed:
            t = torch.tensor([loss_sum, float(n_sum)], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            loss_sum = float(t[0].item())
            n_sum = int(t[1].item())

        train_time_sec = max(time.time() - t_train0, 1e-9)

        metrics_val = {}
        metrics_val_calib = None
        calib_ab = None
        if is_rank0:
            eval_model = ema_model if (ema_enabled and ema_model is not None) else (model.module if isinstance(model, DDP) else model)
            pred_val, true_val = _eval_predictions(eval_model, val_loader, device)
            metrics_val = compute_metrics(pred_val, true_val)
            a, b = fit_affine_calibration(pred_val, true_val)
            calib_ab = (float(a), float(b))
            pred_val_cal = apply_affine(pred_val, a, b)
            metrics_val_calib = compute_metrics(pred_val_cal, true_val)

            key = save_best_key.split("/", 1)[1]
            cur = float(metrics_val[key])
            improved = False
            if best_metric is None:
                improved = True
            elif best_mode == "max" and cur > float(best_metric):
                improved = True
            elif best_mode == "min" and cur < float(best_metric):
                improved = True
            if improved:
                best_metric = cur
                save_checkpoint(
                    best_path,
                    model=eval_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    cfg=cfg,
                    best_metric=best_metric,
                    extra={
                        **(
                            {}
                            if (not ema_enabled or ema_model is None)
                            else {"model_raw": (model.module if isinstance(model, DDP) else model).state_dict(), "ema": {"decay": float(ema_decay)}}
                        ),
                        "val_metrics": metrics_val,
                        "val_calib": (
                            None
                            if (metrics_val_calib is None or calib_ab is None)
                            else {"a": float(calib_ab[0]), "b": float(calib_ab[1]), **{k: float(v) for k, v in metrics_val_calib.items()}}
                        ),
                    },
                )

            rec = {
                "epoch": epoch,
                "time_sec": round(time.time() - t0, 3),
                "train/time_sec": round(train_time_sec, 3),
                "train/steps": int(num_steps),
                "train/updates": int(num_updates),
                "train/examples": int(n_sum),
                "train/examples_per_sec": float(n_sum) / float(train_time_sec),
                "train/loss": loss_sum / max(n_sum, 1),
                "train/max_grad_norm": float(max_grad_norm),
                "train/nan_inf_steps": int(nan_inf_steps),
                "lr": lr,
            }
            if torch.cuda.is_available() and device.type == "cuda":
                rec["cuda/peak_alloc_gb"] = float(torch.cuda.max_memory_allocated(device)) / 1e9
                rec["cuda/peak_reserved_gb"] = float(torch.cuda.max_memory_reserved(device)) / 1e9
            rec.update({f"val/{k}": float(v) for k, v in metrics_val.items()})
            if metrics_val_calib is not None and calib_ab is not None:
                rec.update({f"val_calib/{k}": float(v) for k, v in metrics_val_calib.items()})
                rec["val_calib/a"] = float(calib_ab[0])
                rec["val_calib/b"] = float(calib_ab[1])
            logger.log(rec)  # type: ignore[union-attr]
            print(rec)

        if distributed:
            dist.barrier()

    # Final: evaluate test with best checkpoint
    if is_rank0:
        best_epoch = None
        best_val = None
        best_val_calib = None
        if best_path.exists():
            ckpt = load_checkpoint(best_path, map_location="cpu")
            best_epoch = int(ckpt.get("epoch", -1))
            best_val = ckpt.get("val_metrics", None)
            best_val_calib = ckpt.get("val_calib", None)
            model_eval = _build_model(ckpt["cfg"], state_dict=ckpt["model"]).to(device)
            model_eval.load_state_dict(ckpt["model"], strict=True)
        else:
            model_eval = model.module if isinstance(model, DDP) else model

        pred_test, true_test = _eval_predictions(model_eval, test_loader, device)
        metrics_test = compute_metrics(pred_test, true_test)

        calib_a = None
        calib_b = None
        metrics_test_calib = None
        if isinstance(best_val_calib, dict) and ("a" in best_val_calib) and ("b" in best_val_calib):
            calib_a = float(best_val_calib["a"])
            calib_b = float(best_val_calib["b"])
            metrics_test_calib = compute_metrics(apply_affine(pred_test, calib_a, calib_b), true_test)

        test_out: Dict[str, Any] = {**{k: float(v) for k, v in metrics_test.items()}}
        if metrics_test_calib is not None and calib_a is not None and calib_b is not None:
            test_out["calib"] = {"a": float(calib_a), "b": float(calib_b), **{k: float(v) for k, v in metrics_test_calib.items()}}
        (run_dir / "test_metrics.json").write_text(json.dumps(test_out, indent=2, sort_keys=True) + "\n")

        summary = {
            "run_dir": str(run_dir),
            "task": str(task),
            "seed": int(seed),
            "world_size": int(world),
            "batch_per_gpu": int(batch_size),
            "grad_accum": 1,
            "global_batch": int(batch_size) * int(world),
            "precision": "bf16_autocast" if torch.cuda.is_available() else "fp32_cpu",
            "optim": {"name": "adamw", "lr": float(optim_cfg["lr"]), "weight_decay": float(optim_cfg.get("weight_decay", 0.0))},
            "sched": {"name": str(sched_name), "warmup_epochs": int(warmup_epochs), "min_lr": float(min_lr)},
            "selection": {
                "best_by": str(save_best_key),
                "best_mode": str(best_mode),
                "best_epoch": best_epoch,
                "val@best_raw": best_val,
                "val@best_calib": best_val_calib,
            },
            "results": {"test_raw": metrics_test, "test_calib": metrics_test_calib, "calib_params": {"a": calib_a, "b": calib_b}},
            "artifacts": {"ckpt_best": str(best_path), "config_snapshot": str(run_dir / "config_snapshot.yaml")},
            "init_from": None if args.init_from is None else str(args.init_from),
            "init_from_scope": None if args.init_from is None else str(args.init_from_scope),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print({"test": metrics_test, "test_calib": metrics_test_calib, "best_epoch": best_epoch, "run_dir": str(run_dir)})

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
