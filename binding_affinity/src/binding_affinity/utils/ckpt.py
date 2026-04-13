from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from binding_affinity.utils.paths import resolve_ws_path


def _strip_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    if not any(k.startswith("module.") for k in state.keys()):
        return state
    return {k[len("module.") :]: v for k, v in state.items()}


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    cfg: Dict[str, Any],
    best_metric: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = resolve_ws_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj: Dict[str, Any] = {
        "epoch": int(epoch),
        "cfg": cfg,
        "model": model.state_dict(),
        "best_metric": best_metric,
    }
    if optimizer is not None:
        obj["optimizer"] = optimizer.state_dict()
    if extra:
        obj.update(extra)
    torch.save(obj, path)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    path = resolve_ws_path(path)
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(obj, dict) or "model" not in obj:
        raise ValueError(f"Unexpected checkpoint format at {path}")
    obj["model"] = _strip_module_prefix(obj["model"])
    return obj
