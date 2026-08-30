#!/usr/bin/env python3
"""Mikrobenchmark der 1Cat flash_attn_v100-Kernel auf einer V100.

Gleiches Protokoll wie der Turing-Tile-Sweep (tests/jit_probe/tile_sweep.py
im FA2-Fork): Decode/Verify per q-Skalierung gegen 31k paged KV, Prefill
als 2048er-Chunk. Zwei Geometrien:
  A) 27B-Produktionsgeometrie  H=4  HK=1  D=128 (per-GPU, TP2)
  B) Flash-Next-Geometrie      HK=1 D=256, q_per_kv 6 und 8 (XQA-Gate)

Verify laeuft auf Volta als "smallq tokens-as-batch": k+1 Zeilen mit
gestaffelten seq_lens gegen dieselbe block_table (so baut es der Backend).

Aufruf (auf einer freien V100, CUDA_VISIBLE_DEVICES setzen!):
  .venv-sm70-130/bin/python tools/volta_attn_bench.py
"""

import torch
import flash_attn_v100 as fav

DEV = "cuda:0"
SEQLEN = 31488          # wie Turing-Sweep: ~31k KV
CHUNK = 2048            # Prefill-Chunk
BLOCK = 16              # V100-Backend-Default
ITERS = 50
WARMUP = 10


def make_cache(num_blocks: int, hk: int, d: int):
    k = torch.randn(num_blocks, BLOCK, hk, d, device=DEV, dtype=torch.float16)
    v = torch.randn_like(k)
    return k, v


def bench(fn, iters=ITERS):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def decode_verify_suite(name: str, h: int, hk: int, d: int, use_xqa: bool):
    num_blocks = (SEQLEN + BLOCK - 1) // BLOCK + 8
    k_cache, v_cache = make_cache(num_blocks, hk, d)
    fn_kernel = (fav.flash_attn_decode_paged_xqa if use_xqa
                 else fav.flash_attn_decode_paged)
    results = {}
    for ql in (1, 2, 8):  # 1=Decode, 2/8=Verify tokens-as-batch
        bt = torch.arange(num_blocks - 8, device=DEV, dtype=torch.int32)
        block_table = bt.unsqueeze(0).repeat(ql, 1).contiguous()
        seq_lens = torch.tensor(
            [SEQLEN + i for i in range(ql)], device=DEV, dtype=torch.int32)
        q = torch.randn(ql, h, d, device=DEV, dtype=torch.float16)
        out = torch.empty_like(q)
        try:
            ms = bench(lambda: fn_kernel(
                q, k_cache, v_cache, block_table, seq_lens, out=out))
            results[ql] = ms
        except Exception as exc:  # Gate-Verletzung o.ae. sichtbar machen
            results[ql] = f"FEHLER: {type(exc).__name__}: {exc}"
    del k_cache, v_cache
    torch.cuda.empty_cache()
    line = "  ".join(
        f"q{ql}={ms:.3f}" if isinstance(ms, float) else f"q{ql}={ms}"
        for ql, ms in results.items())
    print(f"{name:44s} {line}", flush=True)


def prefill_suite(name: str, h: int, hk: int, d: int, splitkv: bool):
    total = SEQLEN + CHUNK
    num_blocks = (total + BLOCK - 1) // BLOCK + 8
    k_cache, v_cache = make_cache(num_blocks, hk, d)
    bt = torch.arange(num_blocks - 8, device=DEV, dtype=torch.int32)
    block_table = bt.unsqueeze(0).contiguous()
    seq_lens = torch.tensor([total], device=DEV, dtype=torch.int32)
    q = torch.randn(1, CHUNK, h, d, device=DEV, dtype=torch.float16)
    out = torch.empty_like(q)
    fn_kernel = (fav.flash_attn_prefill_paged_splitkv if splitkv
                 else fav.flash_attn_prefill_paged)
    try:
        ms = bench(lambda: fn_kernel(
            q, k_cache, v_cache, block_table, seq_lens, out=out, causal=True),
            iters=10)
        print(f"{name:44s} chunk{CHUNK}={ms:.2f} ms", flush=True)
    except Exception as exc:
        print(f"{name:44s} FEHLER: {type(exc).__name__}: {exc}", flush=True)
    del k_cache, v_cache
    torch.cuda.empty_cache()


def main():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} (sm{props.major}{props.minor})", flush=True)
    print(f"KV {SEQLEN}, Block {BLOCK}, {ITERS} Iterationen\n", flush=True)

    print("=== A) 27B-Geometrie H=4 HK=1 D=128 ===", flush=True)
    decode_verify_suite("decode_paged (27B-Pfad)", 4, 1, 128, use_xqa=False)
    decode_verify_suite("decode_paged_xqa (Gate-Test)", 4, 1, 128, use_xqa=True)
    prefill_suite("prefill_paged", 4, 1, 128, splitkv=False)
    prefill_suite("prefill_paged_splitkv", 4, 1, 128, splitkv=True)

    print("\n=== B) Flash-Next-Geometrie HK=1 D=256 ===", flush=True)
    for qpk in (6, 8):
        decode_verify_suite(f"decode_paged      D256 H{qpk}", qpk, 1, 256, False)
        decode_verify_suite(f"decode_paged_xqa  D256 H{qpk}", qpk, 1, 256, True)
    prefill_suite("prefill_paged         D256 H8", 8, 1, 256, splitkv=False)
    prefill_suite("prefill_paged_splitkv D256 H8", 8, 1, 256, splitkv=True)


if __name__ == "__main__":
    main()
