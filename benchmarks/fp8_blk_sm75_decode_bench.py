"""sm75-Decode-A/B fuer die dequant16-Linears: fp16-cuBLAS (heutige
sm75-Produktionsroute) gegen die gepackten QPN8-blk-Kernel.

Hintergrund (Handover 2026-09-03 14:00): die RTX-Stufen zahlen ~7 ms/Step
in fp16-Tiny-GEMMs, weil die FP8-blk-Gewichte auf sm75 beim Laden
dequantisiert werden. Kandidat: gemm_qpn8_blk_wmma (WMMA 16x16x16, laeuft
nativ auf Turing) direkt aus dem gepackten Layout — gleiche Dequant-
Mathematik, 1/2 Traffic.

Aufruf: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<rtx> \
  CUDA_HOME=<cuda-12.8> VLLM_SKINNY_NVFP4_SRC=kernels/skinny_kernels.cu \
  .venv-sm70-130/bin/python benchmarks/fp8_blk_sm75_decode_bench.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
from qpn8_blk_test import qpn8_prepack  # noqa: E402

from vllm.model_executor.kernels.linear.nvfp4.marlin import (  # noqa: E402
    _get_skinny_ext,
)

SHAPES = [
    ("wkv", 512, 4096),
    ("wq_a", 1024, 4096),
    ("wq_b", 32768, 1024),
    ("wo_a", 8192, 4096),
    ("wo_b", 4096, 8192),
]
MS = [1, 5, 6, 8]
BN = BK = 128


def time_it(fn, iters=100):
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


def main():
    ext = _get_skinny_ext()
    dev = torch.device("cuda:0")
    print("device:", torch.cuda.get_device_name(0), flush=True)
    print("%-6s %2s | %9s %9s %9s | %8s | rel_err fp16/wmma/qpn8"
          % ("shape", "M", "fp16_ms", "wmma_ms", "qpn8_ms", "best"))
    sums = {"fp16": 0.0, "wmma": 0.0, "qpn8": 0.0}
    for name, n, k in SHAPES:
        torch.manual_seed(0)
        w8 = torch.randint(0, 256, (n, k), dtype=torch.uint8, device=dev)
        w8[w8 == 0x7F] = 0x7E
        w8[w8 == 0xFF] = 0xFE
        wq = w8.view(torch.float8_e4m3fn)
        nb, kb = (n + BN - 1) // BN, (k + BK - 1) // BK
        bscale = (2.0 ** (torch.rand(nb, kb, device=dev) * 4.0 - 12.0)).float()
        sc_full = (bscale.repeat_interleave(BN, 0)[:n]
                   .repeat_interleave(BK, 1)[:, :k])
        ref_w = (wq.float() * sc_full).half()  # heutige sm75-Route (persistent)
        packed = qpn8_prepack(w8)
        splitk, nacc = (16, 3) if (k // 16) % 16 == 0 else (8, 3)
        for m in MS:
            x = (torch.randn(m, k, device=dev, dtype=torch.half) * 0.1)
            y_ref = (x.float() @ ref_w.float().t()).half()

            def fp16_fn():
                return torch.nn.functional.linear(x, ref_w)

            def wmma_fn():
                return ext.gemm_qpn8_blk_wmma(x, packed, bscale, n, BN, BK)

            def qpn8_fn():
                return ext.gemm_qpn8_blk(x, packed, bscale, n, BN, BK,
                                         splitk, nacc)

            errs = []
            outs = {}
            for tag, fn in (("fp16", fp16_fn), ("wmma", wmma_fn),
                            ("qpn8", qpn8_fn)):
                try:
                    y = fn()
                    e = ((y.float() - y_ref.float()).norm()
                         / (y_ref.float().norm() + 1e-30)).item()
                    outs[tag] = time_it(fn)
                    errs.append(f"{e:.1e}")
                except Exception as exc:  # noqa: BLE001 - Kernel darf fehlen
                    outs[tag] = float("nan")
                    errs.append(type(exc).__name__[:10])
            best = min(outs, key=lambda t: outs[t] if outs[t] == outs[t]
                       else 1e9)
            for t, v in outs.items():
                if v == v:
                    sums[t] += v
            print("%-6s %2d | %9.4f %9.4f %9.4f | %8s | %s"
                  % (name, m, outs["fp16"], outs["wmma"], outs["qpn8"],
                     best, "/".join(errs)), flush=True)
    print("Summen (ms ueber alle Zellen):",
          {t: round(v, 3) for t, v in sums.items()})


if __name__ == "__main__":
    main()
