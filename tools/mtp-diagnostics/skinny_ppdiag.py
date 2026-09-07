# SPDX-License-Identifier: Apache-2.0
"""Temporary fork diagnostics (v100-skinny): per-rank tracing of the PP
speculative-decode handoff.

Enabled by VLLM_SKINNY_PPDIAG=<path prefix>. Each rank appends to
``<prefix>.rank<N>.log`` so the four ranks do not interleave. Every tag is
capped so a long run cannot fill the disk. This file is diagnostic scaffolding
and is removed once the defect it chases is fixed.
"""

import os
from collections import defaultdict
from typing import Any

_PREFIX = os.environ.get("VLLM_SKINNY_PPDIAG")
_CAP = int(os.environ.get("VLLM_SKINNY_PPDIAG_CAP", "40"))
_counts: dict[str, int] = defaultdict(int)
_handle = None
_step = 0


def enabled() -> bool:
    return bool(_PREFIX)


def _out():
    global _handle
    if _handle is None:
        try:
            import torch.distributed as dist

            rank = dist.get_rank() if dist.is_initialized() else -1
        except Exception:
            rank = -1
        path = f"{_PREFIX}.rank{rank}.pid{os.getpid()}.log"
        _handle = open(path, "a", buffering=1)
    return _handle


def _fmt(value: Any) -> str:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            flat = value.detach().flatten()
            head = flat[:24].tolist()
            return f"shape={tuple(value.shape)} dtype={value.dtype} {head}"
    except Exception:
        pass
    return repr(value)


def mark(tag: str, **fields: Any) -> None:
    if not _PREFIX:
        return
    # Reading tensors forces a device sync, which CUDA forbids while a graph is
    # being captured. The capture pass runs the same code paths, so skip it.
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
    except Exception:
        return
    _counts[tag] += 1
    if _counts[tag] > _CAP:
        return
    parts = " ".join(f"{k}={_fmt(v)}" for k, v in fields.items())
    _out().write(f"[{_counts[tag]:03d}] step={_step} {tag} {parts}\n")


def bump_step() -> None:
    global _step
    _step += 1


# --- Zeigererfassung (sync-frei, laeuft auch waehrend der Graph-Aufzeichnung) ---
#
# mark() muss sich waehrend des Capture abschalten, weil es Tensorwerte liest
# und damit synchronisiert. data_ptr() synchronisiert nicht, also darf ptrs()
# auch im Capture laufen. Damit laesst sich der Vertrag pruefen, auf dem der
# FULL-Graph-Replay beruht: die Laufzeit muss IN DIESELBEN Puffer schreiben,
# die beim Capture eingebacken wurden.

_MAX_DEPTH = 4


def _walk(obj: Any, path: str, out: list[str], depth: int = 0) -> None:
    import torch

    if depth > _MAX_DEPTH:
        return
    if isinstance(obj, torch.Tensor):
        loc = "cpu" if obj.device.type == "cpu" else "gpu"
        out.append(
            f"{path}=@{obj.data_ptr():#x}/{loc}{tuple(obj.shape)}/{obj.dtype}".replace(
                "torch.", ""
            )
        )
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            _walk(value, f"{path}.{key}", out, depth + 1)
        return
    if isinstance(obj, (list, tuple)):
        # NamedTuples nach Feldnamen, sonst nach Position.
        names = getattr(obj, "_fields", None)
        for i, value in enumerate(obj):
            label = names[i] if names else str(i)
            _walk(value, f"{path}.{label}", out, depth + 1)
        return
    if hasattr(obj, "__dataclass_fields__"):
        for name in obj.__dataclass_fields__:
            _walk(getattr(obj, name, None), f"{path}.{name}", out, depth + 1)
        return
    if isinstance(obj, (int, float, bool)) or obj is None:
        out.append(f"{path}={obj}")


def ptrs(tag: str, **fields: Any) -> None:
    """Protokolliert Datenzeiger statt Werten - erlaubt waehrend des Capture."""
    if not _PREFIX:
        return
    _counts[tag] += 1
    if _counts[tag] > _CAP:
        return
    out: list[str] = []
    for key, value in fields.items():
        _walk(value, key, out)
    _out().write(f"[{_counts[tag]:03d}] step={_step} {tag} " + " ".join(out) + "\n")
