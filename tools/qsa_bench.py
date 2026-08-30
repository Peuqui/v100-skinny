#!/usr/bin/env python3
"""Profil-Sweep fuer den QSA-Sparse-Attention-Kernel (Flash-Next-Grundmodell).

Die Dispatch-Tabelle in qwen4_exp/nvidia/ops/qsa.py ist "Tuned on GB300"
— dieser Harness misst dieselben Regime auf sm70/sm75 und sweept
(BLOCK_N, NUM_SPLITS, num_warps) um die Blackwell-Profile herum.

Geometrie = Flash-Next-Produktion bei TP2: 12 Q-Koepfe, 1 KV-Kopf,
D=256, TOPK=2048 (indexer_budget), PAGE_SIZE=16, bf16-Caches.

Aufruf:  CUDA_VISIBLE_DEVICES=<gpu> .venv-sm70-130/bin/python tools/qsa_bench.py
"""

import torch
import triton
# Pre-Ampere (sm70/75) nutzt in Produktion den amd/-Zweig (reines Triton,
# siehe qwen4_exp/__init__.py) — NICHT nvidia/!
from vllm.models.qwen4_exp.amd.ops.qsa import (
    _qsa_merge_splitk_kernel,
    _qsa_sparse_paged_gqa_splitk_kernel,
    qsa_sparse_paged_attention,
)

DEV = "cuda:0"
HEADS, KV_HEADS, D = 12, 1, 256
TOPK = 2048
PAGE = 16
KV_TOKENS = 16384          # Cache-Fuellstand (Langkontext im 16k-Fenster)
ITERS = 50


def make_cache():
    num_pages = KV_TOKENS // PAGE + 8
    k_cache = torch.randn(num_pages, PAGE, KV_HEADS, D, device=DEV,
                          dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    return k_cache, v_cache, num_pages


def make_inputs(rows: int, k_cache, num_pages):
    q = torch.randn(rows, HEADS, D, device=DEV, dtype=torch.bfloat16)
    # Jede Query waehlt TOPK zufaellige sichtbare Tokens (sortiert, wie
    # der Indexer sie liefert)
    sel = torch.stack([
        torch.randperm(KV_TOKENS, device=DEV)[:TOPK].sort().values
        for _ in range(min(rows, 8))
    ])
    if rows > 8:  # Zeilen 9+ teilen sich Auswahlmuster (Bench, nicht Numerik)
        sel = sel[torch.arange(rows, device=DEV) % 8]
    logical_indices = sel.to(torch.int32).contiguous()
    block_table = torch.arange(num_pages - 8, device=DEV,
                               dtype=torch.int32).unsqueeze(0).contiguous()
    token_to_req = torch.zeros(rows, device=DEV, dtype=torch.int32)
    return q, logical_indices, block_table, token_to_req


def launch(q, k_cache, v_cache, logical_indices, block_table, token_to_req,
           block_n, target_splits, warps):
    out = torch.empty_like(q)
    block_m = triton.next_power_of_2(HEADS // KV_HEADS)
    num_tiles = triton.cdiv(TOPK, block_n)
    max_useful = 1 << (num_tiles.bit_length() - 1)
    num_splits = min(max_useful, target_splits)
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        partial_output = torch.empty((num_splits, *q.shape),
                                     dtype=torch.float32, device=DEV)
        partial_lse = torch.empty((num_splits, q.shape[0], q.shape[1]),
                                  dtype=torch.float32, device=DEV)
    grid = (q.shape[0], KV_HEADS, num_splits)
    _qsa_sparse_paged_gqa_splitk_kernel[grid](
        q, k_cache, v_cache, logical_indices, block_table, token_to_req,
        partial_output, partial_lse, out,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        logical_indices.stride(0), block_table.stride(0),
        out.stride(0), out.stride(1),
        q.shape[0], k_cache.shape[0], block_table.shape[0],
        TOPK=TOPK, PAGE_SIZE=PAGE, PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=HEADS // KV_HEADS, HEAD_DIM=D, NUM_QUERY_HEADS=HEADS,
        NUM_SPLITS=num_splits, NUM_TILES=num_tiles,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=warps, num_stages=2,
    )
    if num_splits > 1:
        _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
            partial_output, partial_lse, out,
            out.stride(0), out.stride(1), q.shape[0],
            HEAD_DIM=D, NUM_QUERY_HEADS=HEADS, NUM_SPLITS=num_splits,
            BLOCK_SPLITS=triton.next_power_of_2(num_splits),
            num_warps=2, num_stages=1,
        )
    return out


def bench(fn, iters=ITERS):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    torch.manual_seed(7)
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} (sm{props.major}{props.minor}) | "
          f"H{HEADS}/KV{KV_HEADS}/D{D}, TOPK {TOPK}, KV {KV_TOKENS}\n",
          flush=True)
    k_cache, v_cache, num_pages = make_cache()

    # Regime: (Zeilen, Name, Produktionsprofil laut GB300-Tabelle)
    regimes = [
        (1, "decode q1", (16, 64, 4)),
        (5, "verify k4 (q5)", (16, 64, 4)),
        (2048, "prefill chunk", (64, 1, 2)),
    ]
    # Sweep-Kandidaten: (block_n, target_splits, warps)
    candidates = [
        (64, 1, 2),    # Produktionsprofil (Referenz)
        (16, 1, 4),    # zurueckgenommener Prefill-Kandidat
        (16, 4, 4),
    ]
    for rows, name, prod in regimes:
        q, li, bt, ttr = make_inputs(rows, k_cache, num_pages)
        ref = qsa_sparse_paged_attention(q, k_cache, v_cache, li, bt, ttr)
        print(f"=== {name} (rows={rows}, Produktionsprofil {prod}) ===",
              flush=True)
        results = []
        for bn, ts, w in candidates:
            try:
                o = launch(q, k_cache, v_cache, li, bt, ttr, bn, ts, w)
                torch.cuda.synchronize()
                err = (o.float() - ref.float()).abs().max().item()
                print(f"  N{bn:<4}S{ts:<4}W{w}: max-abs-Abweichung {err:.3e}",
                      flush=True)
                if err > 2e-2:
                    continue
                ms = bench(lambda: launch(q, k_cache, v_cache, li, bt, ttr,
                                          bn, ts, w),
                           iters=ITERS if rows < 100 else 10)
                tag = " <= PROD" if (bn, ts, w) == prod else ""
                results.append((ms, bn, ts, w))
                print(f"  N{bn:<4}S{ts:<4}W{w}: {ms:8.3f} ms{tag}",
                      flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  N{bn:<4}S{ts:<4}W{w}: FEHLER "
                      f"{type(exc).__name__}: {str(exc)[:80]}", flush=True)
        if results:
            best = min(results)
            print(f"  BESTE: N{best[1]} S{best[2]} W{best[3]} = "
                  f"{best[0]:.3f} ms\n", flush=True)


if __name__ == "__main__":
    main()
