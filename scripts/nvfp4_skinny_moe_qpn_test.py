#!/usr/bin/env python3
"""Grouped MoE QPN kernel (skinny_kernels.cu: moe_qpn, mma.m8n8k4 on
per-expert prepacked fragments) against moe_simt and the per-expert loop on
REAL ModelOpt expert weights (DeepSeek-V4-Flash, layer 5): correctness (max
abs diff vs the loop, fp16 rounding expected) and per-layer timing at the
decode/verify shapes (tokens <= 8), with a SPLITK/NACC sweep per weight
matrix to pick the production config. Run per card with CUDA_VISIBLE_DEVICES
and CUDA_DEVICE_ORDER=PCI_BUS_ID.
Usage: VLLM_SKINNY_NVFP4_SRC=kernels/skinny_kernels.cu python scripts/nvfp4_skinny_moe_qpn_test.py
"""
import time

import torch
from safetensors import safe_open

from vllm.model_executor.kernels.linear.nvfp4.marlin import (
    _get_skinny_ext,
    _qpn_prepack,
)
from vllm.model_executor.layers.fused_moe.experts.nvfp4_skinny_moe import (
    skinny_moe_forward,
)
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
g13_t = torch.tensor(g13, dtype=torch.float32, device="cuda")
g2_t = torch.tensor(g2, dtype=torch.float32, device="cuda")
hidden = w13_c.size(2) * 2
inter = w13_c.size(1) // 2
ext = _get_skinny_ext()
act = lambda out, inp: swiglu_limit_func(out, inp, 10.0)
print("device:", torch.cuda.get_device_name(), "| ext moe_qpn:", hasattr(ext, "moe_qpn"))

# One-time per-expert QPN prepack (pure byte permutation of checkpoint bytes).
t0 = time.perf_counter()
p13 = [_qpn_prepack(w13_c[e], w13_su[e]) for e in range(E)]
p2 = [_qpn_prepack(w2_c[e], w2_su[e]) for e in range(E)]
qc13 = torch.stack([p[0] for p in p13]).contiguous()
qs13 = torch.stack([p[1] for p in p13]).contiguous()
qc2 = torch.stack([p[0] for p in p2]).contiguous()
qs2 = torch.stack([p[1] for p in p2]).contiguous()
del p13, p2
torch.cuda.synchronize()
print(f"prepack: {time.perf_counter() - t0:.1f} s for {E} experts x 2 matrices")
# Checkpoint-shaped views of the packs: what the serving route holds after
# its in-place re-permutation (process_weights_after_loading).
qc13v = qc13.view(E, 2 * inter, hidden // 2)
qs13v = qs13.view(E, 2 * inter, hidden // 16)
qc2v = qc2.view(E, hidden, inter // 2)
qs2v = qs2.view(E, hidden, inter // 16)


def routing(ids):
    """Beide Formate: per-Experte-Offsets (moe_simt) + compact (moe_qpn)."""
    T, topk = ids.shape
    flat = ids.reshape(-1).to(torch.int64)
    S = flat.numel()
    order = torch.argsort(flat, stable=True).to(torch.int32)
    counts = torch.bincount(flat, minlength=E).to(torch.int32)
    offsets = torch.zeros(E + 1, dtype=torch.int32, device="cuda")
    offsets[1:] = torch.cumsum(counts, 0)
    sorted_e = flat[order.long()]
    newg = torch.ones(S, dtype=torch.bool, device="cuda")
    newg[1:] = sorted_e[1:] != sorted_e[:-1]
    gidx = torch.cumsum(newg, 0) - 1
    goff = torch.full((S + 1,), S, dtype=torch.int64, device="cuda")
    goff.scatter_reduce_(0, gidx, torch.arange(S, dtype=torch.int64, device="cuda"),
                         reduce="amin")
    gids = torch.zeros(S, dtype=torch.int64, device="cuda")
    gids.scatter_(0, gidx, sorted_e)
    return order, offsets, gids.to(torch.int32), goff.to(torch.int32)


def grouped_simt(hs, ids, w):
    T, topk = ids.shape
    order, offsets, _, _ = routing(ids)
    y13 = torch.empty(T * topk, 2 * inter, dtype=torch.float16, device="cuda")
    ext.moe_simt(hs, w13_c, w13_su, g13_t, order, offsets, topk, y13, False, T)
    mid = torch.empty(T * topk, inter, dtype=torch.float16, device="cuda")
    act(mid, y13)
    y2 = torch.empty(T * topk, hidden, dtype=torch.float16, device="cuda")
    ext.moe_simt(mid, w2_c, w2_su, g2_t, order, offsets, topk, y2, True, T)
    return (y2.view(T, topk, hidden).float() * w.unsqueeze(-1)).sum(1).half()


def grouped_qpn(hs, ids, w, cfg13, cfg2):
    T, topk = ids.shape
    order, _, gids, goff = routing(ids)
    y13 = torch.empty(T * topk, 2 * inter, dtype=torch.float16, device="cuda")
    ext.moe_qpn(hs, qc13, qs13, g13_t, order, gids, goff, topk, y13, False, T, *cfg13)
    mid = torch.empty(T * topk, inter, dtype=torch.float16, device="cuda")
    act(mid, y13)
    y2 = torch.empty(T * topk, hidden, dtype=torch.float16, device="cuda")
    ext.moe_qpn(mid, qc2, qs2, g2_t, order, gids, goff, topk, y2, True, T, *cfg2)
    return (y2.view(T, topk, hidden).float() * w.unsqueeze(-1)).sum(1).half()


def loop(hs, ids, w):
    """The serving prefill path (gemm_qpn_simt/gemm_qpn on packs)."""
    out = torch.empty(hs.size(0), hidden, dtype=torch.float16, device="cuda")
    skinny_moe_forward(ext=ext, output=out, hidden_states=hs, w1=qc13v, w2=qc2v,
        w1_scales_u8=qs13v, w2_scales_u8=qs2v, g1=g13, g2=g2,
        topk_weights=w, topk_ids=ids, inter_dim=inter, activation_fn=act)
    return out


def t(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


CFGS = [(8, 1), (8, 2), (16, 1), (16, 2), (32, 1), (32, 2)]

# --- Correctness across shapes: grouped_simt on the untouched checkpoint
# layout is the layout-independent reference; both pack readers (moe_qpn
# and the qpn prefill loop) must agree with it ---
ok = True
for T, topk, seed in ((1, 6, 1), (6, 6, 2), (8, 6, 3), (6, 6, 4), (3, 6, 5)):
    torch.manual_seed(seed)
    hs = (torch.randn(T, hidden, device="cuda", dtype=torch.float16) / 8).contiguous()
    _, ids = torch.topk(torch.randn(T, E, device="cuda"), topk, dim=-1)
    ids = ids.to(torch.int32)
    w = torch.rand(T, topk, device="cuda") + 0.1
    a = grouped_simt(hs, ids, w)
    ref = a.float().abs().max().item()
    worst = (a.float() - loop(hs, ids, w).float()).abs().max().item()
    for cfg in CFGS:
        b = grouped_qpn(hs, ids, w, cfg, cfg)
        worst = max(worst, (a.float() - b.float()).abs().max().item())
    good = worst <= 2e-3 * max(ref, 1e-3) + 1e-3
    ok &= good
    print(f"T={T} topk={topk} experts={len(set(ids.flatten().tolist()))}: "
          f"max|diff|={worst:.2e} over loop+{len(CFGS)} cfgs (ref max {ref:.2f}) "
          f"{'OK' if good else 'FAIL'}")

# --- Duplicate-slot multipass: force one expert past 8 rows ---
torch.manual_seed(7)
T, topk = 8, 6
hs = (torch.randn(T, hidden, device="cuda", dtype=torch.float16) / 8).contiguous()
ids = torch.zeros(T, topk, dtype=torch.int32, device="cuda")
ids[:, 0] = 3  # expert 3 gets 8+ slots -> multi-pass
ids[:, 1:] = torch.topk(torch.randn(T, E - 1, device="cuda"), topk - 1, dim=-1)[1].to(torch.int32) + 1
w = torch.rand(T, topk, device="cuda") + 0.1
a = grouped_simt(hs, ids, w)
b = grouped_qpn(hs, ids, w, (16, 2), (16, 2))
err = (a.float() - b.float()).abs().max().item()
ref = a.float().abs().max().item()
good = err <= 2e-3 * max(ref, 1e-3) + 1e-3
ok &= good
print(f"multipass (expert 3 x {T} slots): max|diff|={err:.2e} {'OK' if good else 'FAIL'}")

# --- Timing sweep at the production point T=6 ---
torch.manual_seed(2)
T, topk = 6, 6
hs = (torch.randn(T, hidden, device="cuda", dtype=torch.float16) / 8).contiguous()
_, ids = torch.topk(torch.randn(T, E, device="cuda"), topk, dim=-1)
ids = ids.to(torch.int32)
w = torch.rand(T, topk, device="cuda") + 0.1
order, offsets, gids, goff = routing(ids)
y13 = torch.empty(T * topk, 2 * inter, dtype=torch.float16, device="cuda")
mid = (torch.randn(T * topk, inter, device="cuda", dtype=torch.float16) / 8).contiguous()
y2 = torch.empty(T * topk, hidden, dtype=torch.float16, device="cuda")

print(f"\n-- per-matrix sweep (T={T}, {len(set(ids.flatten().tolist()))} active experts) --")
print(f"simt  w13 {t(lambda: ext.moe_simt(hs, w13_c, w13_su, g13_t, order, offsets, topk, y13, False, T)):.3f} ms   "
      f"w2 {t(lambda: ext.moe_simt(mid, w2_c, w2_su, g2_t, order, offsets, topk, y2, True, T)):.3f} ms")
best13, best2 = None, None
for cfg in CFGS:
    t13 = t(lambda: ext.moe_qpn(hs, qc13, qs13, g13_t, order, gids, goff, topk, y13, False, T, *cfg))
    t2 = t(lambda: ext.moe_qpn(mid, qc2, qs2, g2_t, order, gids, goff, topk, y2, True, T, *cfg))
    print(f"qpn {cfg} w13 {t13:.3f} ms   w2 {t2:.3f} ms")
    if best13 is None or t13 < best13[1]:
        best13 = (cfg, t13)
    if best2 is None or t2 < best2[1]:
        best2 = (cfg, t2)
print(f"best: w13 {best13[0]} {best13[1]:.3f} ms, w2 {best2[0]} {best2[1]:.3f} ms")

print("\n-- full layer (routing + w13 + act + w2 + combine) --")
for T2, seed in ((1, 1), (6, 2), (8, 3)):
    torch.manual_seed(seed)
    hs2 = (torch.randn(T2, hidden, device="cuda", dtype=torch.float16) / 8).contiguous()
    _, ids2 = torch.topk(torch.randn(T2, E, device="cuda"), topk, dim=-1)
    ids2 = ids2.to(torch.int32)
    w2w = torch.rand(T2, topk, device="cuda") + 0.1
    ts = t(lambda: grouped_simt(hs2, ids2, w2w))
    tq = t(lambda: grouped_qpn(hs2, ids2, w2w, best13[0], best2[0]))
    print(f"T={T2}: simt {ts:.3f} ms   qpn {tq:.3f} ms   ({ts / tq:.2f}x)")

print("ALL OK" if ok else "FAILURES")
