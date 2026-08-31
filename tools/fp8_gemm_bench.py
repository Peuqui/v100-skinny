#!/usr/bin/env python3
"""Vergleich der beiden FP8-GEMM-Wege des Forks auf sm70.

`fp8_gemm_sm70_prefill_dispatch_out` ist der schnelle Weg, aber in
fp8.py hinter einem Tor: `tp_size == 4` UND eine der vier fest
verdrahteten Matrixformen. Auf dieser Maschine ist homogenes TP4
unmoeglich (2 RTX + 3 V100), das Tor also unerreichbar.

Die Frage ist damit: liegt es am KERNEL oder nur am TOR? Der Kernel
bekommt k_ld/q_ld als Parameter — er koennte allgemeiner sein, als das
Tor zulaesst. Dieser Bench ruft beide Wege direkt auf, mit den
TP4-Formen (Tabellenfall) und den TP2-Formen (unser Fall).

Aufruf: CUDA_VISIBLE_DEVICES=<v100> fp8_gemm_bench.py
"""
import torch
from vllm import _sm70_ops as sm70_ops

DEV = "cuda:0"
BLOCK = 128          # weight_scale-Blockgroesse wie im Aufrufer
ITERS = 30

# (Name, K, N) — K = Eingangsdimension, N = Ausgangsdimension
SHAPES = [
    ("gate_up TP4 (Tabelle)", 5120, 8704),
    ("down    TP4 (Tabelle)", 4352, 5120),
    ("o_proj  TP4 (Tabelle)", 1536, 5120),
    ("gate_up TP2 (unser)",   5120, 17408),
    ("down    TP2 (unser)",   8704, 5120),
    ("o_proj  TP2 (unser)",   3072, 5120),
]
M = 4096             # Prefill-Chunk, ueber der Schwelle 3920


def make(k, n):
    x = torch.randn(M, k, device=DEV, dtype=torch.float16)
    # Der Kernel nimmt die FP8-Bytes als uint8 entgegen (view, kein cast) —
    # er interpretiert sie selbst als e4m3.
    w = torch.randn(n, k, device=DEV, dtype=torch.float16).to(
        torch.float8_e4m3fn).view(torch.uint8)
    scale = torch.rand(
        (n + BLOCK - 1) // BLOCK, (k + BLOCK - 1) // BLOCK,
        device=DEV, dtype=torch.float16) * 0.02 + 0.01
    out = torch.empty(M, n, device=DEV, dtype=torch.float16)
    return x, w, scale, out


def bench(fn, iters=ITERS):
    for _ in range(5):
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
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} (sm{p.major}{p.minor}) | M={M}, Block={BLOCK}\n",
          flush=True)
    print(f"{'Form':26} {'allgemein':>11} {'schnell':>11} {'Faktor':>8}")
    ws = torch.empty(5120 * 17408, device=DEV, dtype=torch.float16)
    for name, k, n in SHAPES:
        x, w, scale, out = make(k, n)
        try:
            t_gen = bench(lambda: sm70_ops.fp8_gemm_sm70_out(
                out, x, w, scale, BLOCK, k, k))
        except Exception as exc:
            print(f"{name:26} FEHLER {type(exc).__name__}: {str(exc)[:60]}")
            continue
        try:
            ref = out.clone()
            t_fast = bench(lambda: sm70_ops.fp8_gemm_sm70_prefill_dispatch_out(
                out, ws.data_ptr(), x, w, scale, BLOCK, k, k, False, 3920))
            err = (out.float() - ref.float()).abs().max().item()
            note = "" if err < 5e-2 else f"  ABWEICHUNG {err:.1e}"
            print(f"{name:26} {t_gen:9.2f}ms {t_fast:9.2f}ms "
                  f"{t_gen/t_fast:7.2f}x{note}")
        except Exception as exc:
            print(f"{name:26} {t_gen:9.2f}ms  schnell: {type(exc).__name__}: "
                  f"{str(exc)[:50]}")
        del x, w, scale, out
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
