#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "binding_affinity" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from binding_affinity.data.collate import collate_lba_samples  # noqa: E402
from binding_affinity.data.processed_dataset import ProcessedLBADataset  # noqa: E402
from binding_affinity.models.affinity.encoder_faenet import FAENetConfig  # noqa: E402
from binding_affinity.models.affinity.energy_head import EnergyHeadConfig  # noqa: E402
from binding_affinity.models.affinity.model import AffinityModel, AffinityModelConfig  # noqa: E402
from binding_affinity.models.affinity.regressor import RegressorConfig  # noqa: E402
from binding_affinity.models.affinity.tokenizer_hier import TokenizerConfig  # noqa: E402
from binding_affinity.utils.config import load_config  # noqa: E402
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a forward-only sanity check on the copied LBA60 data.")
    parser.add_argument("--config", type=str, default="configs/lba60_quick.yaml")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="train")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    task = str(cfg["task"]["name"])
    processed_root = resolve_ws_path(cfg["io"]["processed_root"])
    ds = ProcessedLBADataset(processed_root / f"{task}_{args.split}.pt")

    batch_size = max(1, min(int(args.batch_size), len(ds)))
    samples = [ds[idx] for idx in range(batch_size)]
    batch = collate_lba_samples(samples)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = _build_model(cfg).to(device)
    model.eval()
    for key in ["complex", "protein", "ligand"]:
        batch[key] = batch[key].to(device)

    with torch.no_grad():
        pred, debug = model(batch)

    payload = {
        "task": task,
        "split": args.split,
        "device": str(device),
        "batch_size": batch_size,
        "ids": batch["id"],
        "pred_shape": list(pred.shape),
        "pred_sample": [float(x) for x in pred.detach().float().cpu().tolist()],
        "debug_keys": sorted(debug.keys()),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
