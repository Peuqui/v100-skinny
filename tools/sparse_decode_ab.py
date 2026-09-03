#!/usr/bin/env python3
"""Standalone-A/B: DSV4-Sparse-Decode — ROCm-Ragged-Kernel (Produktion) vs
die brachliegenden sm70-Paged-FP8-Kernel, auf einer Karte.

Hintergrund (Handover 2026-09-03): der sm70-Impl ist im PP-Verbund nicht
aktivierbar (Backend-Heterogenitaet bricht den Metadata-Vertrag), also
wird hier ZUERST standalone gemessen, ob die Kernel den Umbau lohnen.

Geometrie = DSV4-Flash-Produktion: b Token (K=5-Spekulation -> 6),
64 Heads, D=512 (448 NoPE fp8 + 64 RoPE), SWA-Fenster 128 (main),
Indexer-TopK 512 (extra), Block 256, packed-uint8-Cache. Beide Kernel
lesen DIESELBEN Cache-Bytes/Indizes -> Ausgaben muessen uebereinstimmen.

Aufruf: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
  CUDA_HOME=<cuda-12.8> .venv-sm70-130/bin/python tools/sparse_decode_ab.py
"""
import statistics

import torch
import triton  # noqa: F401

from vllm.models.deepseek_v4.sm70.sparse_kernels import (
    sm70_sparse_attention_paged_fp8,
    sm70_sparse_attention_paged_fp8_splitk,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _rocm_sparse_attn_decode_ragged_triton,
)

DEV = "cuda:0"
HEADS, NOPE, ROPE = 64, 448, 64
D = NOPE + ROPE
BLOCK = 256
SWA_LEN, TOPK = 128, 512
KV_TOKENS = 4096
SCALE = D ** -0.5
ITERS = 50

print("device:", torch.cuda.get_device_name(0), flush=True)
torch.manual_seed(23)

num_blocks = KV_TOKENS // BLOCK + 2
# Zeilenbreite aus der Produktions-Cache-Geometrie: head_bytes deckt
# 448 fp8-Bytes + 64 RoPE-fp16 (+ Skalen) ab. Wir lesen sie nicht aus der
# Config, sondern nehmen die groesste Adresse, die die Kernel anfassen:
# beide adressieren NoPE-Bytes [0..448) und RoPE-fp16 bei [448..448+128).
HEAD_BYTES = NOPE + 2 * ROPE

# Bytes auf endliche fp8-e4m3-/fp16-Muster beschraenken (kein 0x7C..0x7F
# High-Byte -> keine NaN/Inf in beiden Interpretationen).
raw = torch.randint(0, 0x77, (num_blocks, BLOCK, HEAD_BYTES),
                    device=DEV, dtype=torch.uint8)
main_cache = raw
extra_cache = raw  # Produktion: getrennte Caches; fuer das A/B egal.


def make_rows(b):
    torch.manual_seed(100 + b)
    q = torch.randn(b, HEADS, D, device=DEV, dtype=torch.float16)
    sink = torch.randn(HEADS, device=DEV, dtype=torch.float32) * 0.1
    main_idx = torch.stack([
        torch.randperm(KV_TOKENS, device=DEV)[:SWA_LEN].sort().values
        for _ in range(b)
    ]).to(torch.int32)
    extra_idx = torch.stack([
        torch.randperm(KV_TOKENS, device=DEV)[:TOPK].sort().values
        for _ in range(b)
    ]).to(torch.int32)
    main_len = torch.full((b,), SWA_LEN, device=DEV, dtype=torch.int32)
    extra_len = torch.full((b,), TOPK, device=DEV, dtype=torch.int32)
    return q, sink, main_idx, extra_idx, main_len, extra_len


def run_ragged(q, sink, main_idx, extra_idx, b):
    main_indptr = torch.arange(0, (b + 1) * SWA_LEN, SWA_LEN,
                               device=DEV, dtype=torch.int32)
    extra_indptr = torch.arange(0, (b + 1) * TOPK, TOPK,
                                device=DEV, dtype=torch.int32)
    return _rocm_sparse_attn_decode_ragged_triton(
        q, main_cache, main_idx.reshape(-1), main_indptr, SCALE, sink,
        NOPE, ROPE, extra_cache, extra_idx.reshape(-1), extra_indptr)


def run_sm70(q, sink, main_idx, extra_idx, main_len, extra_len):
    out = torch.empty_like(q)
    sm70_sparse_attention_paged_fp8(
        q, main_cache, main_idx, main_len, SCALE, sink, out,
        extra_cache, extra_idx.view(q.shape[0], 1, -1), extra_len)
    return out


def run_sm70_splitk(q, sink, main_idx, extra_idx, main_len, extra_len):
    b = q.shape[0]
    num_partials = (SWA_LEN + 15) // 16 + (TOPK + 15) // 16
    pm = torch.full((b, HEADS, num_partials), float("-inf"),
                    device=DEV, dtype=torch.float32)
    ps = torch.zeros(b, HEADS, num_partials, device=DEV, dtype=torch.float32)
    pa = torch.zeros(b, HEADS, num_partials, D, device=DEV,
                     dtype=torch.float32)
    out = torch.empty_like(q)
    sm70_sparse_attention_paged_fp8_splitk(
        q, main_cache, main_idx, main_len, SCALE, sink, out,
        extra_cache, extra_idx.view(b, 1, -1), extra_len, pm, ps, pa)
    return out


def timeit(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


for b in (1, 6):
    q, sink, mi, ei, ml, el = make_rows(b)
    o_ragged = run_ragged(q, sink, mi, ei, b)
    o_sm70 = run_sm70(q, sink, mi, ei, ml, el)
    o_split = run_sm70_splitk(q, sink, mi, ei, ml, el)
    splitk_ok = True
    for tag, o in (("sm70", o_sm70), ("splitk", o_split)):
        finite = bool(torch.isfinite(o.float()).all())
        if not finite:
            print(f"b={b} Numerik {tag:6s}: NICHT-FINIT (Pfad defekt, "
                  "wird nicht gewertet)", flush=True)
            if tag == "splitk":
                splitk_ok = False
            continue
        d = (o.float() - o_ragged.float()).abs().max().item()
        rel = ((o.float() - o_ragged.float()).norm()
               / (o_ragged.float().norm() + 1e-30)).item()
        print(f"b={b} Numerik {tag:6s} vs ragged: max={d:.3e} rel={rel:.3e}",
              flush=True)
    t_r = timeit(lambda: run_ragged(q, sink, mi, ei, b))
    t_s = timeit(lambda: run_sm70(q, sink, mi, ei, ml, el))
    t_k = timeit(lambda: run_sm70_splitk(q, sink, mi, ei, ml, el)) if splitk_ok else float("nan")
    print(f"b={b}  ragged={t_r:.3f}ms  sm70={t_s:.3f}ms  "
          f"splitk={t_k:.3f}ms  speedup-sm70={t_r / t_s:.2f}x",
          flush=True)
print("DONE", flush=True)
