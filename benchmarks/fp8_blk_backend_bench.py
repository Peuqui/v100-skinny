"""TurboMind W8A16 vs QPN8-blk on block-scaled FP8, real DeepSeek shapes.

Decides whether a per-shape backend chooser is needed at all: if one
backend dominates every (shape, M) cell, a default flip suffices (KISS).
Correctness cross-check per cell against an fp16 dequant reference.

Run inside .venv-sm70 on a V100 (CUDA_VISIBLE_DEVICES to one sm70 card).
"""
import os
import sys

import torch

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")

from vllm import _sm70_ops as sm70_ops  # noqa: E402
from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E402
    process_fp8_weight_block_strategy,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
from qpn8_blk_test import qpn8_prepack  # noqa: E402  (same prepack SSOT)

# DeepSeek-V4-Flash attention linears (full, un-sharded), block [128,128]
SHAPES = [
    ("wkv", 512, 4096),
    ("wq_a", 1024, 4096),
    ("wq_b", 32768, 1024),
    ("wo_a", 8192, 4096),
    ("wo_b", 4096, 8192),
]
MS = [1, 2, 4, 8, 12, 16, 32, 64, 128, 256, 512, 2048]
WMMA_MAX = int(os.environ.get("WMMA_MAX", "64"))
BN = BK = 128
CHUNK_MAX = 96


def time_it(fn, iters):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    beg = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    beg.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return beg.elapsed_time(end) / iters


def qpn8_call(ext, x, packed, bscale, n, bn, bk, splitk, nacc):
    m = x.shape[0]
    if m <= 8:
        return ext.gemm_qpn8_blk(x, packed, bscale, n, bn, bk, splitk, nacc)
    if m <= 16:
        return ext.gemm_qpn8_blk_mt2(x, packed, bscale, n, bn, bk,
                                     min(splitk, 16), nacc)
    if m <= WMMA_MAX:
        # mid band: tensor-core tiles straight from the packed layout
        return ext.gemm_qpn8_blk_wmma(x, packed, bscale, n, bn, bk)
    # large band: fast transient dequant + cuBLAS hgemm
    wf = ext.qpn8_blk_dequant(packed, bscale, n, x.shape[1], bn, bk)
    return torch.nn.functional.linear(x, wf)


def main():
    from vllm.model_executor.kernels.linear.nvfp4.marlin import _get_skinny_ext
    ext = _get_skinny_ext()
    dev = torch.device("cuda:0")
    print("%-6s %6s | %10s %10s | %8s | %s"
          % ("shape", "M", "tm_ms", "qpn8_ms", "speedup", "rel_err(tm/qpn8)"))
    wins = {"tm": 0, "qpn8": 0}
    for name, n, k in SHAPES:
        torch.manual_seed(0)
        w8 = torch.randint(0, 256, (n, k), dtype=torch.uint8, device=dev)
        w8[w8 == 0x7F] = 0x7E
        w8[w8 == 0xFF] = 0xFE
        wq = w8.view(torch.float8_e4m3fn)
        nb, kb = (n + BN - 1) // BN, (k + BK - 1) // BK
        bscale = (2.0 ** (torch.rand(nb, kb, device=dev) * 4.0 - 12.0)).float()
        # reference weights
        sc_full = (bscale.repeat_interleave(BN, 0)[:n]
                   .repeat_interleave(BK, 1)[:, :k])
        ref_w = (wq.float() * sc_full).half()
        # TurboMind format (same preprocessing as fp8.py)
        tw, tsc = process_fp8_weight_block_strategy(wq, bscale)
        tm_w, tm_s, meta = sm70_ops.fp8_sm70_prepare(
            tw.contiguous(), tsc.to(torch.float32).contiguous(), BN, False)
        k_ld, q_ld = int(meta[0].item()), int(meta[1].item())
        # QPN8 format
        packed = qpn8_prepack(w8)
        splitk, nacc = (16, 3) if (k // 16) % 16 == 0 else (8, 3)
        for m in MS:
            x = (torch.randn(m, k, device=dev, dtype=torch.half) * 0.1)
            y_ref = (x.float() @ ref_w.float().t()).half()
            out = torch.empty((m, n), device=dev, dtype=torch.half)

            def tm_fn():
                sm70_ops.fp8_gemm_sm70_out(out, x, tm_w, tm_s, BN, k_ld,
                                           q_ld, False)

            def q_fn():
                return qpn8_call(ext, x, packed, bscale, n, BN, BK,
                                 splitk, nacc)

            tm_fn()
            e_tm = ((out.float() - y_ref.float()).norm()
                    / y_ref.float().norm()).item()
            y_q = q_fn()
            e_q = ((y_q.float() - y_ref.float()).norm()
                   / y_ref.float().norm()).item()
            iters = 100 if m <= 64 else 20
            t_tm = time_it(tm_fn, iters)
            t_q = time_it(q_fn, iters)
            w = "qpn8" if t_q < t_tm else "tm"
            wins[w] += 1
            print("%-6s %6d | %10.4f %10.4f | %7.2fx | %.1e / %.1e  -> %s"
                  % (name, m, t_tm, t_q, t_tm / t_q, e_tm, e_q, w))
    print("wins:", wins)


if __name__ == "__main__":
    main()
