"""Block-scaled FP8 linear on SM70/SM75: three routes on DeepSeek-V4's real shapes.

  native  1Cat's own QPN8 (fp8_qpn8_prepare_sm70 + fp8_qpn8_dispatch_sm70_out):
          QPN8 for M <= 8, dequant into a dense workspace + cuBLAS above.
  tm      1Cat's TurboMind W8A16 (fp8_sm70_prepare + fp8_gemm_sm70_out).
  skinny  the fork's QPN8-blk routes: M <= 8 QPN8, <= 16 MT2, <= 256 WMMA,
          transient dequant + cuBLAS above.

Shapes are the block-FP8 linears DeepSeek-V4-Flash loads in production (the
QPN8_BLK_CENSUS_LOAD lines of the fork's boot log), block [128, 128].
Every cell is checked against an fp16 dequant reference.

Run with the production venv (fork installed, 1Cat ops + skinny extension):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<one card> \
  VLLM_SKINNY_NVFP4_SRC=/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu \
  /home/mp/vllm/venv/bin/python fp8_blk_three_way.py
"""

import torch

from vllm import _sm70_ops as sm70_ops
from vllm.model_executor.kernels.linear.nvfp4.marlin import _get_skinny_ext
from vllm.model_executor.layers.quantization.modelopt import _sm70_qpn8_prepack
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_weight_block_strategy,
)

SHAPES = [
    ("fused_wqa_wkv", 1536, 4096),
    ("indexer.wq_b", 8192, 1024),
    ("wq_b", 32768, 1024),
    ("wo_b", 4096, 8192),
    ("shared.gate_up", 4096, 4096),
    ("shared.down", 4096, 2048),
]
MS = [1, 2, 4, 6, 8, 12, 16, 32, 64, 128, 256, 512, 2048]
BLOCK = 128
WMMA_MAX = 256


def time_ms(fn, iters: int) -> float:
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


def rel_err(y: torch.Tensor, ref: torch.Tensor) -> float:
    return ((y.float() - ref.float()).norm() / ref.float().norm()).item()


def main() -> None:
    device = torch.device("cuda:0")
    print(torch.cuda.get_device_name(device))
    # TurboMind's FP8 GEMM has no SM75 kernel ("No feasible kernel"), so the
    # tm column is only measured on SM70.
    with_tm = torch.cuda.get_device_capability(device) == (7, 0)
    ext = _get_skinny_ext()
    header = "%-15s %5s | %8s %8s %8s | %s"
    print(header % ("shape", "M", "native", "tm", "skinny", "rel err n/t/s"))
    for name, n, k in SHAPES:
        torch.manual_seed(0)
        raw = torch.randint(0, 256, (n, k), dtype=torch.uint8, device=device)
        raw[raw == 0x7F] = 0x7E
        raw[raw == 0xFF] = 0xFE
        weight = raw.view(torch.float8_e4m3fn)
        scales = (
            2.0 ** (torch.rand(n // BLOCK, k // BLOCK, device=device) * 4.0 - 12.0)
        ).float()
        full = scales.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
        ref_w = (weight.float() * full).half()

        codes, group_scales = sm70_ops.fp8_qpn8_prepare_sm70(weight, scales)
        dense = torch.empty((n, k), dtype=torch.float16, device=device)
        split_k = 8 if k % 256 else 16

        if with_tm:
            tm_w, tm_s = process_fp8_weight_block_strategy(weight, scales)
            tm_weight, tm_scales, meta = sm70_ops.fp8_sm70_prepare(
                tm_w.contiguous(), tm_s.float().contiguous(), BLOCK, False
            )
            k_ld, q_ld = int(meta[0].item()), int(meta[1].item())

        packed = _sm70_qpn8_prepack(raw)
        sk_split = 16 if (k // 16) % 16 == 0 else 8

        for m in MS:
            x = torch.randn(m, k, device=device, dtype=torch.half) * 0.1
            ref = (x.float() @ ref_w.float().t()).half()
            out_native = torch.empty((m, n), device=device, dtype=torch.half)
            out_tm = torch.empty((m, n), device=device, dtype=torch.half)

            def native() -> None:
                sm70_ops.fp8_qpn8_dispatch_sm70_out(
                    out_native,
                    dense.data_ptr(),
                    x,
                    codes,
                    group_scales,
                    split_k,
                    2,
                    False,
                    False,
                )

            def tm() -> None:
                sm70_ops.fp8_gemm_sm70_out(
                    out_tm, x, tm_weight, tm_scales, BLOCK, k_ld, q_ld, False
                )

            def skinny() -> torch.Tensor:
                if m <= 8:
                    return ext.gemm_qpn8_blk(
                        x, packed, scales, n, BLOCK, BLOCK, sk_split, 3
                    )
                if m <= 16:
                    return ext.gemm_qpn8_blk_mt2(
                        x, packed, scales, n, BLOCK, BLOCK, min(sk_split, 16), 3
                    )
                if m <= WMMA_MAX:
                    return ext.gemm_qpn8_blk_wmma(x, packed, scales, n, BLOCK, BLOCK)
                dequant = ext.qpn8_blk_dequant(packed, scales, n, k, BLOCK, BLOCK)
                return torch.nn.functional.linear(x, dequant)

            native()
            y_skinny = skinny()
            iters = 100 if m <= 64 else 20
            tm_ms = float("nan")
            tm_err = float("nan")
            if with_tm:
                tm()
                tm_err = rel_err(out_tm, ref)
                tm_ms = time_ms(tm, iters)
            print(
                "%-15s %5d | %8.4f %8.4f %8.4f | %.1e %.1e %.1e"
                % (
                    name,
                    m,
                    time_ms(native, iters),
                    tm_ms,
                    time_ms(skinny, iters),
                    rel_err(out_native, ref),
                    tm_err,
                    rel_err(y_skinny, ref),
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
