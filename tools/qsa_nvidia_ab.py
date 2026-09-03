#!/usr/bin/env python3
"""A/B fuer den UPSTREAM-nvidia-QSA-Kernel (origin/main) auf sm70/sm75.

Gate fuer den geplanten Upstream-PR (Gate <80 + smem-Clamp, Issue #441 /
PR #455-Nachgang): unsere schmalen N16-Profile gegen die GB300-Tabelle
(N64@W2) und den sm70-Retune (N64/N32@W4) — gemessen mit der ORIGINALEN
nvidia-Datei, nur die Profil-Funktion wird je Messpunkt uebersteuert.

Geometrie = Flash-Next TP2: 12 Q-Koepfe, 1 KV-Kopf, D=256, TOPK=2048,
PAGE=16, bf16. base_programs = rows (1 KV-Kopf).

Aufruf:
  CUDA_VISIBLE_DEVICES=<gpu> CUDA_HOME=<cuda-12.8> \
    .venv-sm70-130/bin/python tools/qsa_nvidia_ab.py <pfad/zu/upstream_qsa.py>
"""
import importlib.util
import statistics
import sys

import torch
import triton  # noqa: F401  (Kernel-Compile)

QSA_PATH = sys.argv[1]
spec = importlib.util.spec_from_file_location("qsa_upstream", QSA_PATH)
qsa = importlib.util.module_from_spec(spec)
sys.modules["qsa_upstream"] = qsa
spec.loader.exec_module(qsa)

DEV = "cuda:0"
NAME = torch.cuda.get_device_name(0)
CAP = torch.cuda.get_device_capability(0)
HEADS, KV_HEADS, D = 12, 1, 256
TOPK = 2048
PAGE = 16
KV_TOKENS = 16384
ITERS = 30

print(f"device: {NAME} sm{CAP[0]}{CAP[1]}", flush=True)

torch.manual_seed(11)
num_pages = KV_TOKENS // PAGE + 8
k_cache = torch.randn(num_pages, PAGE, KV_HEADS, D, device=DEV,
                      dtype=torch.bfloat16)
v_cache = torch.randn_like(k_cache)
block_table = torch.arange(num_pages - 8, device=DEV,
                           dtype=torch.int32).unsqueeze(0).contiguous()


def make_inputs(rows: int):
    q = torch.randn(rows, HEADS, D, device=DEV, dtype=torch.bfloat16)
    sel = torch.stack([
        torch.randperm(KV_TOKENS, device=DEV)[:TOPK].sort().values
        for _ in range(min(rows, 8))
    ])
    if rows > 8:
        sel = sel[torch.arange(rows, device=DEV) % 8]
    logical_indices = sel.to(torch.int32).contiguous()
    token_to_req = torch.zeros(rows, device=DEV, dtype=torch.int32)
    return q, logical_indices, token_to_req


orig_profile = qsa._qsa_sparse_launch_profile


def timed(inputs, profile):
    """profile: None = Original-Dispatch; sonst (block_n, splits, warps)."""
    if profile is None:
        qsa._qsa_sparse_launch_profile = orig_profile
    else:
        qsa._qsa_sparse_launch_profile = lambda *a, **k: profile
    q, li, ttr = inputs
    out = None
    for _ in range(3):  # Warmup + JIT
        out = qsa.qsa_sparse_paged_attention(
            q, k_cache, v_cache, li, block_table, ttr)
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        qsa.qsa_sparse_paged_attention(q, k_cache, v_cache, li, block_table,
                                       ttr)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times), out


for rows in (1, 6, 8, 64, 256, 512, 1024, 2048):
    base = rows * KV_HEADS
    is70 = CAP == (7, 0)
    configs = {
        "default": None,                      # was die Karte heute bekommt
        "gb300": orig_profile(base, 16, False),   # GB300-Tabelle erzwungen
        "sm70ret": orig_profile(base, 16, True),  # sm70-Retune erzwungen
    }
    if base <= 256:
        configs["ours16"] = (16, 8, 4)
    elif base <= 512:
        configs["ours16"] = (16, 4, 4)
    else:
        configs["ours16"] = (16, 1, 4)

    inputs = make_inputs(rows)
    ref = None
    line = [f"rows={rows:5d}"]
    for tag, prof in configs.items():
        try:
            ms, out = timed(inputs, prof)
        except Exception as exc:  # noqa: BLE001 - Messpunkt kann legal scheitern
            line.append(f"{tag}=FAIL({type(exc).__name__})")
            continue
        if ref is None:
            ref = out
        else:
            md = (out.float() - ref.float()).abs().max().item()
            if md > 5e-2:
                line.append(f"{tag}=NUMERIK! d={md:.2e}")
                continue
        eff = prof if prof is not None else orig_profile(base, 16, is70)
        line.append(f"{tag}[N{eff[0]}/S{eff[1]}/W{eff[2]}]={ms:7.3f}ms")
    print("  ".join(line), flush=True)

qsa._qsa_sparse_launch_profile = orig_profile
print("DONE", flush=True)
