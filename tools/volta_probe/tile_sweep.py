#!/usr/bin/env python3
"""Kachel-Sweep fuer 1Cats Volta-Prefill-Kernel (D=128, paged).

Kompiliert fused_mha_forward_paged.cu pro Kandidat per JIT mit
uebersteuerten BLOCK_M_128/BLOCK_N_128 (Defines sind im Quell-Clone
geguardet). Numerik gegen die Referenzkachel 32x176, Bench wie der
Turing-Sweep: 2048er-Chunk gegen ~31k paged KV, H=4 HK=1 D=128.
"""

import os
import torch
from torch.utils.cpp_extension import load

SRC = "/home/mp/Projekte/1Cat-vLLM/flash-attention-v100"
DEV = "cuda:0"
SEQLEN = 31488
CHUNK = 2048
BLOCK = 16
H, HK, D = 4, 1, 128

# (BLOCK_M, BLOCK_N, smem-Schaetzung Q+K+V in KB)
CANDIDATES = [
    (32, 176, 96),   # Referenz (1Cat)
    (64, 64, 87),    # Runde 1: 15,09 ms — Stabilitaets-Wiederholung
    (64, 80, 95),
    (48, 112, 91),
    (80, 48, 95),
]


def build(bm: int, bn: int):
    return load(
        name=f"volta_probe_m{bm}n{bn}",
        sources=[f"{SRC}/kernel/fused_mha_forward_paged.cu",
                 "/home/mp/Projekte/v100-skinny/tools/volta_probe/shim.cpp"],
        extra_include_paths=[f"{SRC}/include"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70",
                           f"-DPROBE_BM_128={bm}", f"-DPROBE_BN_128={bn}",
                           "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=False,
    )


def make_inputs():
    total = SEQLEN + CHUNK
    num_blocks = (total + BLOCK - 1) // BLOCK + 8
    k_cache = torch.randn(num_blocks, BLOCK, HK, D, device=DEV,
                          dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)
    bt = torch.arange(num_blocks - 8, device=DEV, dtype=torch.int32)
    block_table = bt.unsqueeze(0).contiguous()
    seq_lens = torch.tensor([total], device=DEV, dtype=torch.int32)
    q = torch.randn(1, H, CHUNK, D, device=DEV, dtype=torch.float16)
    return q, k_cache, v_cache, block_table, seq_lens


def main():
    torch.manual_seed(7)
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | KV {SEQLEN}+{CHUNK}, H{H}/HK{HK}/D{D}",
          flush=True)
    q, k_cache, v_cache, block_table, seq_lens = make_inputs()
    scale = D ** -0.5
    ref_out = None
    for bm, bn, smem in CANDIDATES:
        tag = f"M{bm:<3d} N{bn:<3d} (~{smem}KB)"
        try:
            mod = build(bm, bn)
        except Exception as exc:
            print(f"{tag}: BUILD FAILED ({str(exc)[-160:]})", flush=True)
            continue
        try:
            out = mod.prefill(q, k_cache, v_cache, block_table, seq_lens,
                              scale)
            torch.cuda.synchronize()
            if ref_out is None:
                ref_out = out.float()
                err = 0.0
            else:
                err = (out.float() - ref_out).abs().max().item()
            if err > 5e-3:
                print(f"{tag}: NUMERIK-FEHLER max abs {err:.2e}", flush=True)
                continue
            for _ in range(5):
                mod.prefill(q, k_cache, v_cache, block_table, seq_lens, scale)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(10):
                mod.prefill(q, k_cache, v_cache, block_table, seq_lens, scale)
            e.record()
            torch.cuda.synchronize()
            ms = s.elapsed_time(e) / 10
            print(f"{tag}: chunk{CHUNK}={ms:.2f} ms  (num-abw {err:.1e})",
                  flush=True)
        except Exception as exc:
            print(f"{tag}: LAUFZEIT-FEHLER {type(exc).__name__}: "
                  f"{str(exc)[:160]}", flush=True)
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
