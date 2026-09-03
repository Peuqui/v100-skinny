#!/usr/bin/env python3
"""Grouped NVFP4 MoE kernel (skinny_kernels.cu: moe_simt) against the per-expert
loop on REAL ModelOpt expert weights (DeepSeek-V4-Flash, layer 5): correctness
(max abs diff vs the loop, fp16 rounding expected) and per-layer timing at the
decode/verify shapes (tokens <= 8). Run per card with CUDA_VISIBLE_DEVICES and
CUDA_DEVICE_ORDER=PCI_BUS_ID; Triton/nvcc JIT for the local arch on first use.
Usage: VLLM_SKINNY_NVFP4_SRC=kernels/skinny_kernels.cu python scripts/nvfp4_skinny_moe_grouped_test.py
"""
import sys, time, torch
from safetensors import safe_open
from vllm.model_executor.layers.fused_moe.experts.nvfp4_skinny_moe import skinny_moe_forward
from vllm.model_executor.kernels.linear.nvfp4.marlin import _get_skinny_ext, _qpn_prepack
from vllm.model_executor.layers.fused_moe.utils import swiglu_limit_func
shard = "/home/mp/models/DeepSeek-V4-Flash-nvfp4-DSpark/model-00007-of-00046.safetensors"
E = 64
w13_c, w13_s, g13, w2_c, w2_s, g2 = [], [], [], [], [], []
with safe_open(shard, framework="pt", device="cuda") as f:
    for e in range(E):
        b = f"layers.5.ffn.experts.{e}"
        w13_c.append(torch.cat([f.get_tensor(f"{b}.w1.weight"), f.get_tensor(f"{b}.w3.weight")], 0).contiguous())
        w13_s.append(torch.cat([f.get_tensor(f"{b}.w1.weight_scale"), f.get_tensor(f"{b}.w3.weight_scale")], 0).contiguous())
        g13.append(float(f.get_tensor(f"{b}.w1.weight_scale_2").item()))
        w2_c.append(f.get_tensor(f"{b}.w2.weight").contiguous())
        w2_s.append(f.get_tensor(f"{b}.w2.weight_scale").contiguous())
        g2.append(float(f.get_tensor(f"{b}.w2.weight_scale_2").item()))
w13_c, w13_s, w2_c, w2_s = map(torch.stack, (w13_c, w13_s, w2_c, w2_s))
w13_su, w2_su = w13_s.view(torch.uint8), w2_s.view(torch.uint8)
g13_t = torch.tensor(g13, dtype=torch.float32, device="cuda"); g2_t = torch.tensor(g2, dtype=torch.float32, device="cuda")
hidden = w13_c.size(2) * 2; inter = w13_c.size(1) // 2
ext = _get_skinny_ext()
act = lambda out, inp: swiglu_limit_func(out, inp, 10.0)
print("device:", torch.cuda.get_device_name(), "| ext moe_simt:", hasattr(ext, "moe_simt"))

# The serving loop path now reads QPN fragment order (see
# nvfp4_skinny_moe.py); moe_simt below stays on the checkpoint layout.
p13 = [_qpn_prepack(w13_c[e], w13_su[e]) for e in range(E)]
p2 = [_qpn_prepack(w2_c[e], w2_su[e]) for e in range(E)]
qc13v = torch.stack([p[0] for p in p13]).view_as(w13_c).contiguous()
qs13v = torch.stack([p[1] for p in p13]).view_as(w13_su).contiguous()
qc2v = torch.stack([p[0] for p in p2]).view_as(w2_c).contiguous()
qs2v = torch.stack([p[1] for p in p2]).view_as(w2_su).contiguous()
del p13, p2

def grouped(hs, ids, w):
    T, topk = ids.shape
    flat = ids.reshape(-1).to(torch.int32)
    order = torch.argsort(flat, stable=True).to(torch.int32)          # perm: Slots nach Experte
    counts = torch.bincount(flat.to(torch.int64), minlength=E).to(torch.int32)
    offsets = torch.zeros(E + 1, dtype=torch.int32, device="cuda"); offsets[1:] = torch.cumsum(counts, 0)
    y13 = torch.empty(T * topk, 2 * inter, dtype=torch.float16, device="cuda")
    ext.moe_simt(hs, w13_c, w13_su, g13_t, order, offsets, topk, y13, False, T)
    mid = torch.empty(T * topk, inter, dtype=torch.float16, device="cuda")
    act(mid, y13)
    y2 = torch.empty(T * topk, hidden, dtype=torch.float16, device="cuda")
    ext.moe_simt(mid, w2_c, w2_su, g2_t, order, offsets, topk, y2, True, T)
    return (y2.view(T, topk, hidden).float() * w.unsqueeze(-1)).sum(1).half()

def loop(hs, ids, w):
    out = torch.empty(hs.size(0), hidden, dtype=torch.float16, device="cuda")
    skinny_moe_forward(ext=ext, output=out, hidden_states=hs, w1=qc13v, w2=qc2v,
        w1_scales_u8=qs13v, w2_scales_u8=qs2v, g1=g13, g2=g2,
        topk_weights=w, topk_ids=ids, inter_dim=inter, activation_fn=act)
    return out

ok = True
for T, topk, seed in ((1, 6, 1), (6, 6, 2), (8, 6, 3), (6, 6, 4), (3, 6, 5)):
    torch.manual_seed(seed)
    hs = (torch.randn(T, hidden, device="cuda", dtype=torch.float16) / 8).contiguous()
    _, ids = torch.topk(torch.randn(T, E, device="cuda"), topk, dim=-1); ids = ids.to(torch.int32)
    w = torch.rand(T, topk, device="cuda") + 0.1
    a, b = loop(hs, ids, w), grouped(hs, ids, w)
    err = (a.float() - b.float()).abs().max().item(); ref = a.float().abs().max().item()
    good = err <= 2e-3 * max(ref, 1e-3) + 1e-3
    ok &= good
    def t(fn):
        for _ in range(5): fn()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(30): fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t0) / 30 * 1000
    print(f"T={T} topk={topk} experts={len(set(ids.flatten().tolist()))}: max|diff|={err:.2e} (ref max {ref:.2f}) {'OK' if good else 'FAIL'} | loop {t(lambda: loop(hs, ids, w)):.2f} ms  grouped {t(lambda: grouped(hs, ids, w)):.2f} ms")
print("ALL OK" if ok else "FAILURES")
