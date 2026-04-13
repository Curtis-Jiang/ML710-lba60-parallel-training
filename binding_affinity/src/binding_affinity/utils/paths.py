from __future__ import annotations

from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


def repo_root() -> Path:
    # .../binding_affinity/src/binding_affinity/utils/paths.py -> repo root is 4 levels up
    return Path(__file__).resolve().parents[4]


def resolve_ws_path(path: PathLike) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root() / path


def ensure_dir(path: PathLike) -> Path:
    path = resolve_ws_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
