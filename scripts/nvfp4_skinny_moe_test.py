"""Per-expert skinny NVFP4 MoE forward vs. the dequant reference.

Drives ``skinny_moe_forward`` (the compute core of Nvfp4SkinnySm70Experts)
with REAL checkpoint expert weights and compares against the emulation's
own dequantize_to_dtype + a plain torch MoE loop. Both sides share the
same activation function (vLLM's swiglu_limit_func, clamp 10.0, as
DeepSeek-V4-Flash sets), so the comparison isolates routing, the skinny
GEMM dispatch (simt/wmma/64-row chunks) and the weighted reduce.
"""
import argparse
import sys

import torch
from safetensors import safe_open

from vllm.model_executor.layers.fused_moe.experts.nvfp4_skinny_moe import (
    skinny_moe_forward,
)
from vllm.model_executor.layers.fused_moe.utils import swiglu_limit_func
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)

LAYER = 5
CLAMP = 10.0


def load_experts(shard, expert_ids):
    """Stacked (w13, w2) codes/scales/gscales for the given experts."""
    w13_c, w13_s, g13, w2_c, w2_s, g2 = [], [], [], [], [], []
    with safe_open(shard, framework="pt", device="cuda") as f:
        for e in expert_ids:
            base = f"layers.{LAYER}.ffn.experts.{e}"
            # checkpoint keeps gate (w1) and up (w3) separate; vLLM loads
            # them fused as w13 = [gate; up].
            c1 = f.get_tensor(f"{base}.w1.weight_packed")
            c3 = f.get_tensor(f"{base}.w3.weight_packed")
            s1 = f.get_tensor(f"{base}.w1.weight_scale")
            s3 = f.get_tensor(f"{base}.w3.weight_scale")
            g1 = float(f.get_tensor(f"{base}.w1.weight_global_scale").item())
            g3 = float(f.get_tensor(f"{base}.w3.weight_global_scale").item())
            if g1 != g3:
                print(f"  note: expert {e} w1/w3 gscale differ "
                      f"({g1} vs {g3}); using w1's (matches vLLM)")
            w13_c.append(torch.cat([c1, c3], dim=0).contiguous())
            w13_s.append(torch.cat([s1, s3], dim=0).contiguous())
            g13.append(1.0 / g1)
            w2_c.append(f.get_tensor(f"{base}.w2.weight_packed").contiguous())
            w2_s.append(f.get_tensor(f"{base}.w2.weight_scale").contiguous())
            g2.append(
                1.0 / float(
                    f.get_tensor(f"{base}.w2.weight_global_scale").item()))
    return (torch.stack(w13_c), torch.stack(w13_s), g13,
            torch.stack(w2_c), torch.stack(w2_s), g2)


def dequant(codes, scales, gscale):
    return dequantize_to_dtype(
        tensor_fp4=codes,
        tensor_sf=scales,
        global_scale=torch.tensor(gscale, device="cuda", dtype=torch.float32),
        dtype=torch.float16,
        block_size=16,
        swizzle=False,
    )


def act(out, inp):
    swiglu_limit_func(out, inp, CLAMP)


def run(bundle, num_tokens, topk, forced_expert, seed, label):
    w13_c, w13_s, g13, w2_c, w2_s, g2 = bundle
    num_experts = w13_c.size(0)
    n13 = w13_c.size(0), w13_c.size(1)
    hidden = w13_c.size(2) * 2
    inter = w13_c.size(1) // 2

    torch.manual_seed(seed)
    hs = (torch.randn(num_tokens, hidden, device="cuda",
                      dtype=torch.float16) / 8).contiguous()
    if forced_expert is not None:
        topk_ids = torch.full((num_tokens, topk), forced_expert,
                              device="cuda", dtype=torch.int32)
    else:
        logits = torch.randn(num_tokens, num_experts, device="cuda")
        _, topk_ids = torch.topk(logits, topk, dim=-1)
        topk_ids = topk_ids.to(torch.int32)
    topk_weights = torch.rand(num_tokens, topk, device="cuda") + 0.1

    from vllm.model_executor.kernels.linear.nvfp4.marlin import (
        _get_skinny_ext,
    )
    got = torch.empty(num_tokens, hidden, device="cuda", dtype=torch.float16)
    skinny_moe_forward(
        ext=_get_skinny_ext(), output=got, hidden_states=hs,
        w1=w13_c, w2=w2_c,
        w1_scales_u8=w13_s.view(torch.uint8),
        w2_scales_u8=w2_s.view(torch.uint8),
        g1=g13, g2=g2,
        topk_weights=topk_weights, topk_ids=topk_ids,
        inter_dim=inter, activation_fn=act,
    )

    ref = torch.zeros(num_tokens, hidden, device="cuda", dtype=torch.float32)
    for t in range(num_tokens):
        for slot in range(topk):
            e = int(topk_ids[t, slot])
            y13 = (hs[t:t + 1].float()
                   @ dequant(w13_c[e], w13_s[e], g13[e]).float().T)
            ia = torch.empty(1, inter, device="cuda", dtype=torch.float16)
            act(ia, y13.half())
            y = ia.float() @ dequant(w2_c[e], w2_s[e], g2[e]).float().T
            ref[t] += float(topk_weights[t, slot]) * y[0]

    denom = ref.abs().max().clamp(min=1e-3)
    rel = (got.float() - ref).abs().max() / denom
    ok = bool(rel < 5e-3)
    print(f"{label:24s} M={num_tokens:4d} topk={topk} "
          f"rel_err={rel:.2e}  {'OK' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    args = ap.parse_args()
    print(f"device={torch.cuda.get_device_name()}")

    experts = list(range(12))
    bundle = load_experts(args.shard, experts)

    cases = [
        # (tokens, topk, forced_expert, seed, label)
        (1, 6, None, 1001, "decode single token"),
        (1, 8, None, 1002, "decode topk=8"),
        (8, 6, None, 1003, "mtp-sized batch"),
        (33, 6, None, 1004, "small prefill"),
        (128, 8, None, 1005, "prefill wmma band"),
        (100, 4, 3, 1006, "forced expert m_e=400 (chunked)"),
        (2, 2, 7, 1007, "forced expert duplicates"),
    ]
    results = [run(bundle, *c) for c in cases]
    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
