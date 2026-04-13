from __future__ import annotations

import os
import sys
from pathlib import Path


def setup_path() -> Path:
    repo = Path(__file__).resolve().parents[2]
    src_root = repo / "binding_affinity" / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return repo


def ensure_cuda_libs_visible() -> None:
    """Ensure NVIDIA driver libs are visible (re-exec if needed).

    Some images do not include `/usr/local/nvidia/lib64` on the default loader path, so
    `torch.cuda.is_available()` can be False even when GPUs exist. Setting `LD_LIBRARY_PATH`
    inside Python is too late (loader path is initialized at process start), so we re-exec
    the current process once with the corrected environment.
    """
    lib_dir = Path("/usr/local/nvidia/lib64")
    if not lib_dir.exists():
        return
    if os.environ.get("BINDING_AFFINITY_CUDA_REEXEC") == "1":
        return
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    lib_str = str(lib_dir)
    if ld.split(":")[0] == lib_str or f":{lib_str}:" in f":{ld}:":
        return
    # Re-exec with updated env so the dynamic loader sees the new search path.
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{lib_str}:{ld}" if ld else lib_str
    env["BINDING_AFFINITY_CUDA_REEXEC"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)
