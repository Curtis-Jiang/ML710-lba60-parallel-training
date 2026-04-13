from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Union

import yaml

from binding_affinity.utils.paths import repo_root


ConfigDict = Dict[str, Any]
PathLike = Union[str, Path]


def _deep_update(base: MutableMapping[str, Any], update: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            _deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def _resolve_include(base_path: Path, include: str) -> Path:
    include_path = Path(include)
    if include_path.is_absolute():
        return include_path
    # Prefer workspace-root relative includes (recommended in repo instructions).
    candidate = repo_root() / include_path
    if candidate.exists():
        return candidate
    # Fallback: relative to current config file.
    candidate = base_path.parent / include_path
    if candidate.exists():
        return candidate
    # Last resort: still return workspace-root relative.
    return repo_root() / include_path


def load_config(path: PathLike) -> ConfigDict:
    path = Path(path)
    if not path.is_absolute():
        path = repo_root() / path
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    includes = raw.pop("includes", []) or []
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, Iterable):
        raise ValueError(f"includes in {path} must be a list[str]")

    cfg: ConfigDict = {}
    for inc in includes:
        inc_path = _resolve_include(path, str(inc))
        _deep_update(cfg, load_config(inc_path))
    _deep_update(cfg, raw)
    return cfg


def set_by_dotted_key(cfg: MutableMapping[str, Any], dotted: str, value: Any) -> None:
    cur: MutableMapping[str, Any] = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value
