from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from _path_setup import ensure_cuda_libs_visible, setup_path

ensure_cuda_libs_visible()
setup_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from binding_affinity.data.collate import collate_lba_samples  # noqa: E402
from binding_affinity.data.processed_dataset import ProcessedLBADataset  # noqa: E402
from binding_affinity.models.affinity.encoder_faenet import FAENetConfig  # noqa: E402
from binding_affinity.models.affinity.energy_head import EnergyHeadConfig  # noqa: E402
from binding_affinity.models.affinity.model import AffinityModel, AffinityModelConfig  # noqa: E402
from binding_affinity.models.affinity.regressor import RegressorConfig  # noqa: E402
from binding_affinity.models.affinity.tokenizer_hier import TokenizerConfig  # noqa: E402
from binding_affinity.utils.ckpt import load_checkpoint  # noqa: E402
from binding_affinity.utils.metrics import apply_affine, compute_metrics, fit_affine_calibration  # noqa: E402
from binding_affinity.utils.paths import resolve_ws_path  # noqa: E402


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


@torch.no_grad()
def _predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
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
    parser = argparse.ArgumentParser(description="Evaluate a binding-affinity checkpoint.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--calibrate_on", type=str, choices=["train", "val"], default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    cfg = ckpt["cfg"]
    task = cfg["task"]["name"]
    processed_root = resolve_ws_path(cfg["io"]["processed_root"])
    ds = ProcessedLBADataset(processed_root / f"{task}_{args.split}.pt")
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate_lba_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg, state_dict=ckpt["model"]).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    pred, true = _predict(model, loader, device)
    metrics_raw = compute_metrics(pred, true)

    out: Dict[str, Any] = {"split": str(args.split), "raw": metrics_raw}

    if args.calibrate_on:
        calib_split = str(args.calibrate_on)
        ds_cal = ProcessedLBADataset(processed_root / f"{task}_{calib_split}.pt")
        loader_cal = DataLoader(
            ds_cal,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_lba_samples,
        )
        pred_c, true_c = _predict(model, loader_cal, device)
        a, b = fit_affine_calibration(pred_c, true_c)
        metrics_calib = compute_metrics(apply_affine(pred, a, b), true)
        out["calib"] = {"calibrate_on": calib_split, "a": float(a), "b": float(b), **metrics_calib}

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
