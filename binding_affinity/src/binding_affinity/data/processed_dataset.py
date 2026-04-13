from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from torch.utils.data import Dataset

from binding_affinity.utils.paths import resolve_ws_path


PathLike = Union[str, Path]


@dataclass(frozen=True)
class ProcessedSplit:
    samples: List[Dict[str, Any]]
    meta: Dict[str, Any]


class ProcessedLBADataset(Dataset[Dict[str, Any]]):
    def __init__(self, processed_path: PathLike):
        processed_path = resolve_ws_path(processed_path)
        obj = torch.load(processed_path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "samples" in obj:
            self.samples = list(obj["samples"])
            self.meta = dict(obj.get("meta") or {})
        elif isinstance(obj, list):
            self.samples = list(obj)
            self.meta = {}
        else:
            raise ValueError(f"Unexpected processed file format at {processed_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[int(idx)]
