#!/usr/bin/env python3
"""moe_qpn on a REAL DeepSeek-V4-Flash layer (MXFP4 original, layer 5, all 256
experts): per-layer time of the grouped kernel (w13 + w2, activation excluded)
at decode and prefill shapes, with effective bandwidth over the expert bytes
the batch touches and effective fp16 throughput. Weights go through the same
path as serving: rebase_e8m0_for_fp16 + _qpn_prepack(sg=32), scale_mode 1.

Run per card (op from the installed 1Cat-vLLM build):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<i> CUDA_HOME=/home/mp/vllm/cuda \\
  python scripts/mxfp4_moe_qpn_roofline.py [--distinct D ...] [tokens ...]

--distinct D routes the decode batch onto exactly D different experts (the
tokens * top_k slots spread over them), to see how the time follows the
weight bytes; without it each token draws top_k random experts.
"""
import argparse

import torch
from safetensors import safe_open

import vllm._C  # noqa: F401  registers torch.ops._C
from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import (
    grouped_splitk,
    qpn_prepack,
    rebase_e8m0_for_fp16,
)

SHARD = ("/home/mp/models/DeepSeek-V4-Flash-284B-A13B-MXFP4-FP8-DSpark/"
         "model-00007-of-00048.safetensors")
LAYER, EXPERTS, TOP_K = 5, 256, 6
parser = argparse.ArgumentParser()
parser.add_argument("tokens", type=int, nargs="*", default=[6, 64, 512, 4096])
parser.add_argument("--distinct", type=int, nargs="*", default=[])
args = parser.parse_args()

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
        qc, qs = qpn_prepack(w[e], s[e], 32)
        w[e].view(-1).copy_(qc)
        s[e].view(-1).copy_(qs)

hidden, n13 = w13.size(2) * 2, w13.size(1)
inter = n13 // 2
bytes_per_expert = (w13[0].numel() + s13[0].numel() + w2[0].numel() + s2[0].numel())
props = torch.cuda.get_device_properties(0)
print(f"device {props.name} sm{props.major}{props.minor} | hidden {hidden} inter {inter} "
      f"| {bytes_per_expert / 2**20:.1f} MiB per expert")


def routing(tokens, seed, distinct=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if distinct:
        # token t takes experts (t * top_k + j) mod D of a random D-subset:
        # every token keeps top_k different experts (D >= top_k) and all D
        # are touched once tokens * top_k >= D
        pool = torch.randperm(EXPERTS, device="cuda", generator=generator)[:distinct]
        offsets = torch.arange(TOP_K, device="cuda")
        ids = torch.stack([pool[(t * TOP_K + offsets) % distinct] for t in range(tokens)])
    else:
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


cases = [(6, d) for d in args.distinct] or [(t, 0) for t in args.tokens]
for tokens, distinct in cases:
    perm, gids, goff, active = routing(tokens, tokens, distinct)
    x = torch.randn(tokens, hidden, dtype=torch.float16, device="cuda")
    mid = torch.randn(tokens * TOP_K, inter, dtype=torch.float16, device="cuda") * 0.1
    y13 = torch.empty(tokens * TOP_K, n13, dtype=torch.float16, device="cuda")
    y2 = torch.empty(tokens * TOP_K, hidden, dtype=torch.float16, device="cuda")

    def run():
        torch.ops._C.skinny_moe_qpn_sm70(x, w13, s13, g13, perm, gids, goff, TOP_K, y13,
                                         False, tokens, grouped_splitk(hidden, 16), 1, 1)
        torch.ops._C.skinny_moe_qpn_sm70(mid, w2, s2, g2, perm, gids, goff, TOP_K, y2,
                                         True, tokens, grouped_splitk(inter, 8), 1, 1)

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
