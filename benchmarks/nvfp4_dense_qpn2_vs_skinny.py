"""Dense NVFP4 on SM70/SM75: 1Cat's QPN2 GEMM against the fork's skinny routes.

Shapes are Qwen3.8-27B's NVFP4 MLP at TP2 (the only dense NVFP4 layers among
the production models). Each path gets its best split-K per M, so neither is
handicapped by a config table. The two independent kernels are checked against
each other.
"""

import torch

from vllm import _sm70_ops as sm70_ops
from vllm.model_executor.kernels.linear.nvfp4.marlin import (
    _get_skinny_ext,
    _qpn_prepack,
)

SHAPES = [("gate_up", 17408, 5120), ("down", 5120, 8704)]
MS = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
SPLITS = [(8, 1), (16, 2), (32, 2)]


def time_ms(fn, iters: int = 100) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) / iters


def main() -> None:
    device = torch.device("cuda:0")
    print(torch.cuda.get_device_name(device))
    ext = _get_skinny_ext()
    gscale = 2.0**-8
    print("%-8s %4s | %8s %8s | %s" % ("shape", "M", "1Cat", "skinny", "rel diff"))
    for name, n, k in SHAPES:
        torch.manual_seed(0)
        codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=device)
        scale_bytes = torch.randint(0x30, 0x40, (n, k // 16), dtype=torch.uint8)
        scales = scale_bytes.to(device).view(torch.float8_e4m3fn)
        cat_codes, cat_scales = sm70_ops.nvfp4_qpn2_prepare_sm70(codes, scales)
        sk_codes, sk_scales = _qpn_prepack(codes, scales.view(torch.uint8))
        for m in MS:
            x = torch.randn(m, k, device=device, dtype=torch.float16) * 0.1
            out = torch.empty((m, n), device=device, dtype=torch.float16)
            best_cat, best_sk = float("inf"), float("inf")
            y_sk = None
            for split, nacc in SPLITS:
                if (k // 16) % split:
                    continue
                if m <= 32:

                    def cat(split=split, nacc=nacc) -> None:
                        sm70_ops.nvfp4_qpn2_gemm_sm70_out(
                            out, x, cat_codes, cat_scales, gscale, split, nacc
                        )

                    best_cat = min(best_cat, time_ms(cat))
                if m <= 8:

                    def sk(split=split, nacc=nacc) -> torch.Tensor:
                        return ext.gemm_qpn2(x, sk_codes, sk_scales, gscale, n, split, nacc)

                    best_sk = min(best_sk, time_ms(sk))
                    y_sk = sk()
            if 8 < m <= 16:
                best_sk = time_ms(lambda: ext.gemm_qpn(x, sk_codes, sk_scales, gscale, n))
                y_sk = ext.gemm_qpn(x, sk_codes, sk_scales, gscale, n)
            elif 16 < m <= 64:
                raw_scales = scales.view(torch.uint8)
                best_sk = time_ms(lambda: ext.gemm_wmma(x, codes, raw_scales, gscale))
                y_sk = ext.gemm_wmma(x, codes, raw_scales, gscale)
            diff = float("nan")
            if m <= 32 and y_sk is not None:
                sm70_ops.nvfp4_qpn2_gemm_sm70_out(
                    out, x, cat_codes, cat_scales, gscale, 16, 2
                )
                diff = ((y_sk.float() - out.float()).norm() / out.float().norm()).item()
            print(
                "%-8s %4d | %8.4f %8.4f | %.1e" % (name, m, best_cat, best_sk, diff),
                flush=True,
            )


if __name__ == "__main__":
    main()
