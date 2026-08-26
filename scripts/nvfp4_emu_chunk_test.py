"""Chunk-sum equivalence for the chunked NVFP4 MoE emulation.

The chunked emulation splits one MoE layer into several expert-parallel
shards and adds the partial sums. This checks that claim on the Triton
expert path directly, with fp16 weights (the dequantization itself is
unchanged upstream code, only WHICH experts get dequantized is new).
"""

import sys

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts


def run(num_experts, num_tokens, topk, hidden, inter, chunk, seed):
    torch.manual_seed(seed)
    dev = "cuda"
    dt = torch.float16
    hs = torch.randn(num_tokens, hidden, device=dev, dtype=dt) / 8
    w1 = torch.randn(num_experts, 2 * inter, hidden, device=dev, dtype=dt) / 16
    w2 = torch.randn(num_experts, hidden, inter, device=dev, dtype=dt) / 16

    logits = torch.randn(num_tokens, num_experts, device=dev, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=-1), topk, dim=-1)
    topk_ids = topk_ids.to(torch.int32)

    full = fused_experts(
        hs, w1, w2, topk_weights, topk_ids,
        activation=MoEActivation.SILU, global_num_experts=num_experts,
    )

    selected = torch.unique(topk_ids.flatten()).to(torch.long)
    acc = torch.zeros_like(full)
    for start in range(0, selected.numel(), chunk):
        ids = selected[start:start + chunk]
        emap = torch.full((num_experts,), -1, dtype=torch.int32, device=dev)
        emap[ids] = torch.arange(ids.numel(), dtype=torch.int32, device=dev)
        part = fused_experts(
            hs, w1[ids].contiguous(), w2[ids].contiguous(), topk_weights, topk_ids,
            activation=MoEActivation.SILU, global_num_experts=num_experts,
            expert_map=emap,
        )
        acc += part

    denom = full.abs().max().clamp(min=1e-6)
    rel = (acc - full).abs().max() / denom
    n_chunks = (selected.numel() + chunk - 1) // chunk
    ok = rel < 3e-3
    print(f"E={num_experts:4d} M={num_tokens:5d} topk={topk} chunk={chunk:3d} "
          f"selected={selected.numel():4d} chunks={n_chunks:3d} "
          f"rel_err={rel:.2e} {'OK' if ok else 'FAIL'}")
    return ok


def main():
    cases = [
        # (experts, tokens, topk, hidden, inter, chunk, seed)
        (256, 1, 6, 512, 256, 4, 1001),     # decode, single token
        (256, 8, 6, 512, 256, 4, 1002),     # decode with MTP-sized batch
        (256, 8, 6, 512, 256, 1, 1003),     # one expert per chunk
        (256, 64, 6, 512, 256, 4, 1004),    # most experts live
        (256, 512, 6, 512, 256, 16, 1005),  # prefill-shaped
        (32, 4, 2, 256, 128, 4, 1006),      # chunk == selected (single pass)
        (32, 128, 8, 256, 128, 32, 1007),   # single chunk covers everything
        (64, 33, 3, 384, 192, 5, 1008),     # ragged chunk tail
    ]
    results = [run(*c) for c in cases]
    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
