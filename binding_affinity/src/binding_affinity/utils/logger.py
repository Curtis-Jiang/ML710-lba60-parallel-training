from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from binding_affinity.utils.paths import resolve_ws_path


@dataclass
class JsonlLogger:
    path: Path

    def __post_init__(self) -> None:
        self.path = resolve_ws_path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
