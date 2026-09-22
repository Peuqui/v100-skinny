#!/usr/bin/env python3
"""Bit-identity harness for moe_qpn changes: runs the grouped kernel on a REAL
DeepSeek-V4-Flash layer (MXFP4 original, layer 5, 256 experts, top-6) with
fixed inputs at decode and prefill shapes, for w13 and w2, and either saves the
outputs (--save FILE) or compares them bit for bit (--compare FILE).

Run per card with the kernel source under test:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<i> CUDA_HOME=/home/mp/vllm/cuda \\
  TORCH_EXTENSIONS_DIR=<own dir> VLLM_SKINNY_NVFP4_SRC=<kernels/skinny_kernels.cu> \\
  python scripts/moe_qpn_bitcheck.py --save|--compare FILE
"""
import sys

import torch
from safetensors import safe_open

from vllm.model_executor.kernels.linear.nvfp4.marlin import _get_skinny_ext, _qpn_prepack
from vllm.model_executor.layers.fused_moe.experts.nvfp4_skinny_moe import (
    _MOE_QPN_CFG,
    rebase_e8m0_for_fp16,
)

SHARD = ("/home/mp/models/DeepSeek-V4-Flash-284B-A13B-MXFP4-FP8-DSpark/"
         "model-00007-of-00048.safetensors")
LAYER, EXPERTS, TOP_K = 5, 256, 6
TOKENS = (1, 6, 64, 512, 4096)
mode, path = sys.argv[1], sys.argv[2]
assert mode in ("--save", "--compare")

with safe_open(SHARD, framework="pt", device="cuda") as f:
    def get(e, name):
        return f.get_tensor(f"layers.{LAYER}.ffn.experts.{e}.{name}").view(torch.uint8)

    w13 = torch.stack([torch.cat([get(e, "w1.weight"), get(e, "w3.weight")]) for e in range(EXPERTS)])
    s13 = torch.stack([torch.cat([get(e, "w1.scale"), get(e, "w3.scale")]) for e in range(EXPERTS)])
    w2 = torch.stack([get(e, "w2.weight") for e in range(EXPERTS)])
    s2 = torch.stack([get(e, "w2.scale") for e in range(EXPERTS)])
g13 = rebase_e8m0_for_fp16(s13).cuda()
g2 = rebase_e8m0_for_fp16(s2).cuda()
for w, s in ((w13, s13), (w2, s2)):
    for e in range(EXPERTS):
        qc, qs = _qpn_prepack(w[e], s[e], 32)
        w[e].view(-1).copy_(qc)
        s[e].view(-1).copy_(qs)
hidden, n13 = w13.size(2) * 2, w13.size(1)
inter = n13 // 2
ext = _get_skinny_ext()
print("device", torch.cuda.get_device_name(0))


def routing(tokens, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.stack([torch.randperm(EXPERTS, device="cuda", generator=generator)[:TOP_K]
                       for _ in range(tokens)])
    flat = ids.reshape(-1).to(torch.int64)
    slots = flat.numel()
    perm = torch.argsort(flat, stable=True).to(torch.int32)
    sorted_e = flat[perm.long()]
    new_group = torch.ones(slots, dtype=torch.bool, device="cuda")
    new_group[1:] = sorted_e[1:] != sorted_e[:-1]
    gidx = torch.cumsum(new_group, 0) - 1
    goff = torch.full((slots + 1,), slots, dtype=torch.int64, device="cuda")
    goff.scatter_reduce_(0, gidx, torch.arange(slots, dtype=torch.int64, device="cuda"), reduce="amin")
    gids = torch.zeros(slots, dtype=torch.int64, device="cuda")
    gids.scatter_(0, gidx, sorted_e)
    return perm, gids.to(torch.int32), goff.to(torch.int32)


results = {}
for tokens in TOKENS:
    perm, gids, goff = routing(tokens, tokens)
    generator = torch.Generator(device="cuda").manual_seed(1000 + tokens)
    x = torch.randn(tokens, hidden, dtype=torch.float16, device="cuda", generator=generator)
    mid = torch.randn(tokens * TOP_K, inter, dtype=torch.float16, device="cuda", generator=generator) * 0.1
    y13 = torch.empty(tokens * TOP_K, n13, dtype=torch.float16, device="cuda")
    y2 = torch.empty(tokens * TOP_K, hidden, dtype=torch.float16, device="cuda")
    ext.moe_qpn(x, w13, s13, g13, perm, gids, goff, TOP_K, y13, False, tokens,
                _MOE_QPN_CFG[0], _MOE_QPN_CFG[1], 1)
    ext.moe_qpn(mid, w2, s2, g2, perm, gids, goff, TOP_K, y2, True, tokens,
                _MOE_QPN_CFG[2], _MOE_QPN_CFG[3], 1)
    results[f"w13_{tokens}"] = y13.cpu()
    results[f"w2_{tokens}"] = y2.cpu()

if mode == "--save":
    torch.save(results, path)
    print("saved", len(results), "outputs to", path)
    sys.exit(0)
reference = torch.load(path)
bad = 0
for name, out in results.items():
    ref = reference[name]
    same = torch.equal(out, ref)
    bad += not same
    diff = (out.float() - ref.float()).abs().max().item()
    print(f"{name:10s}: {'bit-identical' if same else f'DIFF max {diff:.3e}'}")
print("RESULT:", "ALL BIT-IDENTICAL" if bad == 0 else f"{bad} differ")
sys.exit(1 if bad else 0)
