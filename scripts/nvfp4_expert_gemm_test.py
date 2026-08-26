"""Can the existing skinny NVFP4 GEMM serve one MoE expert unchanged?

The chunked emulation dequantizes expert weights to fp16 and hands them to
Triton. The fork already ships a benchmarked NVFP4 GEMM family whose
documented input layout -- codes uint8 [N][K/2] (two e2m1 per byte, low
nibble = even k), scales uint8 [N][K/16] fp8-e4m3, one global scale -- is
exactly what a compressed-tensors NVFP4 MoE checkpoint stores per expert.
If a slice can be fed in unchanged, the decode band needs no grouped MoE
kernel, only a per-expert dispatch.

Uses REAL checkpoint bytes and compares against the emulation's own
dequantize_to_dtype + matmul, with the same global scale on both sides so
the comparison isolates the packing convention.
"""
import argparse
import sys

import torch
from safetensors import safe_open

from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)

LAYER = 5


def load_expert(shard, expert, proj):
    base = f"layers.{LAYER}.ffn.experts.{expert}.{proj}"
    with safe_open(shard, framework="pt", device="cuda") as f:
        codes = f.get_tensor(f"{base}.weight_packed")
        scales = f.get_tensor(f"{base}.weight_scale")
        gscale = f.get_tensor(f"{base}.weight_global_scale")
    # the checkpoint stores the RECIPROCAL global scale (e.g. 21504.0);
    # dequantization multiplies by 1/that.
    return codes.contiguous(), scales.contiguous(), 1.0 / float(gscale.item())


def run(ext, codes, scales, gscale, m, label):
    n, k_half = codes.shape
    k = k_half * 2
    torch.manual_seed(1000 + m)
    x = (torch.randn(m, k, device="cuda", dtype=torch.float16) / 8).contiguous()

    # gemm_qpn wants the fragment-order layout, not the raw checkpoint bytes.
    # _qpn_prepack is a pure permutation and already ships with the fork.
    from vllm.model_executor.kernels.linear.nvfp4.marlin import _qpn_prepack

    qc, qs = _qpn_prepack(codes, scales.view(torch.uint8).contiguous())
    assert qc is not None, "shape not eligible for the qpn prepack"
    got = ext.gemm_qpn(x, qc, qs, gscale, n)

    ref_w = dequantize_to_dtype(
        tensor_fp4=codes,
        tensor_sf=scales,
        global_scale=torch.tensor(gscale, device="cuda", dtype=torch.float32),
        dtype=torch.float16,
        block_size=16,
        swizzle=False,
    )
    ref = x.float() @ ref_w.float().T

    denom = ref.abs().max().clamp(min=1e-3)
    rel = (got.float() - ref).abs().max() / denom
    ok = bool(rel < 5e-3)
    print(f"{label:10s} M={m:3d} N={n:5d} K={k:5d}  rel_err={rel:.2e}  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    args = ap.parse_args()

    from vllm.model_executor.kernels.linear.nvfp4.marlin import _get_skinny_ext

    ext = _get_skinny_ext()
    print(f"device={torch.cuda.get_device_name()}")

    results = []
    for proj in ("w1", "w2"):
        codes, scales, gscale = load_expert(args.shard, 0, proj)
        for m in (1, 2, 6, 8, 16):
            results.append(run(ext, codes, scales, gscale, m, proj))
    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
