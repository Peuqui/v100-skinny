#!/usr/bin/env python3
"""moe_qpn on a REAL DeepSeek-V4-Flash layer (MXFP4 original, layer 5, all 256
experts): per-layer time of the grouped kernel (w13 + w2, activation excluded)
at decode and prefill shapes, with effective bandwidth over the expert bytes
the batch touches and effective fp16 throughput. Weights go through the same
path as serving: rebase_e8m0_for_fp16 + _qpn_prepack(sg=32), scale_mode 1.

Run per card:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<i> CUDA_HOME=/home/mp/vllm/cuda \\
  VLLM_SKINNY_NVFP4_SRC=kernels/skinny_kernels.cu \\
  python scripts/mxfp4_moe_qpn_roofline.py [tokens ...]
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
TOKENS = tuple(int(t) for t in sys.argv[1:]) or (6, 64, 512, 4096)

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
bytes_per_expert = (w13[0].numel() + s13[0].numel() + w2[0].numel() + s2[0].numel())
ext = _get_skinny_ext()
props = torch.cuda.get_device_properties(0)
print(f"device {props.name} sm{props.major}{props.minor} | hidden {hidden} inter {inter} "
      f"| {bytes_per_expert / 2**20:.1f} MiB per expert | cfg {_MOE_QPN_CFG}")


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
    return perm, gids.to(torch.int32), goff.to(torch.int32), int(flat.unique().numel())


for tokens in TOKENS:
    perm, gids, goff, active = routing(tokens, tokens)
    x = torch.randn(tokens, hidden, dtype=torch.float16, device="cuda")
    mid = torch.randn(tokens * TOP_K, inter, dtype=torch.float16, device="cuda") * 0.1
    y13 = torch.empty(tokens * TOP_K, n13, dtype=torch.float16, device="cuda")
    y2 = torch.empty(tokens * TOP_K, hidden, dtype=torch.float16, device="cuda")

    def run():
        ext.moe_qpn(x, w13, s13, g13, perm, gids, goff, TOP_K, y13, False, tokens,
                    _MOE_QPN_CFG[0], _MOE_QPN_CFG[1], 1)
        ext.moe_qpn(mid, w2, s2, g2, perm, gids, goff, TOP_K, y2, True, tokens,
                    _MOE_QPN_CFG[2], _MOE_QPN_CFG[3], 1)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    reps = 20 if tokens <= 512 else 5
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        run()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / reps
    gbps = active * bytes_per_expert / (ms / 1e3) / 1e9
    flops = 2 * tokens * TOP_K * (n13 * hidden + hidden * inter)
    tflops = flops / (ms / 1e3) / 1e12
    print(f"tokens {tokens:5d}: {ms:7.3f} ms  active experts {active:3d}  "
          f"{gbps:6.0f} GB/s over touched weights  {tflops:5.1f} TFLOP/s")
