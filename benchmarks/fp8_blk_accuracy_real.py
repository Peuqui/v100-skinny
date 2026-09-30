"""Accuracy of the block-FP8 routes on real DeepSeek-V4 weights, against fp32.

Layer-3 weights and block scales straight from the checkpoint; activations
with a few outlier channels, as DeepSeek-V4 hidden states have them. Prints
the relative error of 1Cat's native QPN8 dispatch and of the fork's skinny
QPN8-blk routes per M.
"""

import json

import torch
from safetensors import safe_open

from vllm import _sm70_ops as sm70_ops
from vllm.model_executor.kernels.linear.nvfp4.marlin import _get_skinny_ext
from vllm.model_executor.layers.quantization.modelopt import _sm70_qpn8_prepack

ROOT = "/home/mp/models/DeepSeek-V4-Flash-284B-A13B-MXFP4-FP8-DSpark"
NAMES = [
    "layers.3.attn.wq_b",
    "layers.3.attn.wo_b",
    "layers.3.ffn.shared_experts.w2",
]
MS = [1, 6, 8, 12, 20, 32, 64, 256, 512]
BLOCK = 128


def load(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    index = json.load(open(f"{ROOT}/model.safetensors.index.json"))["weight_map"]
    tensors = []
    for suffix in ("weight", "scale"):
        key = f"{name}.{suffix}"
        with safe_open(f"{ROOT}/{index[key]}", "pt") as f:
            tensors.append(f.get_tensor(key))
    return tensors[0], tensors[1].float()


def skinny(ext, x, packed, scales, n, k):
    m = x.shape[0]
    split = 16 if (k // 16) % 16 == 0 else 8
    if m <= 8:
        return ext.gemm_qpn8_blk(x, packed, scales, n, BLOCK, BLOCK, split, 3)
    if m <= 16:
        return ext.gemm_qpn8_blk_mt2(x, packed, scales, n, BLOCK, BLOCK, split, 3)
    if m <= 256:
        return ext.gemm_qpn8_blk_wmma(x, packed, scales, n, BLOCK, BLOCK)
    dense = ext.qpn8_blk_dequant(packed, scales, n, k, BLOCK, BLOCK)
    return torch.nn.functional.linear(x, dense)


def main() -> None:
    device = torch.device("cuda:0")
    print(torch.cuda.get_device_name(device))
    ext = _get_skinny_ext()
    for name in NAMES:
        weight, scales = load(name)
        weight = weight.to(device)
        scales = scales.to(device).contiguous()
        n, k = weight.shape
        full = scales.repeat_interleave(BLOCK, 0)[:n].repeat_interleave(BLOCK, 1)[:, :k]
        ref_w = weight.float() * full
        codes, group_scales = sm70_ops.fp8_qpn8_prepare_sm70(weight, scales)
        dense = torch.empty((n, k), dtype=torch.float16, device=device)
        packed = _sm70_qpn8_prepack(weight.view(torch.uint8))
        torch.manual_seed(1)
        outliers = torch.randperm(k, device=device)[:8]
        for m in MS:
            x = torch.randn(m, k, device=device) * 0.5
            x[:, outliers] *= 60.0
            x16 = x.half()
            ref = x16.float() @ ref_w.t()
            out = torch.empty((m, n), dtype=torch.float16, device=device)
            sm70_ops.fp8_qpn8_dispatch_sm70_out(
                out, dense.data_ptr(), x16, codes, group_scales,
                16 if k % 256 == 0 else 8, 2, False, False,
            )
            y_skinny = skinny(ext, x16, packed, scales, n, k)
            errors = [
                ((y.float() - ref).norm() / ref.norm()).item()
                for y in (out, y_skinny)
            ]
            worst = [
                ((y.float() - ref).abs().max() / ref.abs().max()).item()
                for y in (out, y_skinny)
            ]
            print(
                f"{name:32s} M={m:4d} rel native={errors[0]:.2e} skinny={errors[1]:.2e}"
                f" | max native={worst[0]:.2e} skinny={worst[1]:.2e}",
                flush=True,
            )


if __name__ == "__main__":
    main()
