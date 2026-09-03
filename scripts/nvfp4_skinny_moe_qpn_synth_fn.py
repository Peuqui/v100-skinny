#!/usr/bin/env python3
"""moe_qpn auf der Qwen3.8-Flash-Next-TP1-Geometrie (E=512, topk=10,
w13 [1280,2560], w2 [2560,640]) mit SYNTHETISCHEN NVFP4-Bytes: belegt die
Shape-Generik (Routing/Prepack/Kernel auf voellig anderer Geometrie als
DSv4) und liefert ms/Layer fuer T=1 und T=6. Korrektheits-Anker: moe_simt
auf dem Checkpoint-Layout derselben Bytes. Je Karte via CUDA_VISIBLE_DEVICES."""
import time

import torch

from vllm.model_executor.kernels.linear.nvfp4.marlin import (
    _get_skinny_ext,
    _qpn_prepack,
)

E, TOPK = 512, 10
HID, INTER = 2560, 640
N13, K13 = 2 * INTER, HID    # 1280 x 2560
N2, K2 = HID, INTER          # 2560 x 640
ext = _get_skinny_ext()
print("device:", torch.cuda.get_device_name())

torch.manual_seed(42)
w13_c = torch.randint(0, 256, (E, N13, K13 // 2), dtype=torch.uint8, device="cuda")
w2_c = torch.randint(0, 256, (E, N2, K2 // 2), dtype=torch.uint8, device="cuda")
# fp8-e4m3-Scales in moderatem Bereich (Exponentbits 56..64 ~ 0.5..2.0)
w13_s = torch.randint(52, 60, (E, N13, K13 // 16), dtype=torch.uint8, device="cuda")
w2_s = torch.randint(52, 60, (E, N2, K2 // 16), dtype=torch.uint8, device="cuda")
g13 = torch.full((E,), 0.01, dtype=torch.float32, device="cuda")
g2 = torch.full((E,), 0.01, dtype=torch.float32, device="cuda")

t0 = time.perf_counter()
p13 = [_qpn_prepack(w13_c[e], w13_s[e]) for e in range(E)]
p2 = [_qpn_prepack(w2_c[e], w2_s[e]) for e in range(E)]
qc13 = torch.stack([p[0] for p in p13]).contiguous()
qs13 = torch.stack([p[1] for p in p13]).contiguous()
qc2 = torch.stack([p[0] for p in p2]).contiguous()
qs2 = torch.stack([p[1] for p in p2]).contiguous()
del p13, p2
torch.cuda.synchronize()
print(f"prepack {time.perf_counter()-t0:.1f} s fuer {E} Experten x 2")


def routing(ids):
    flat = ids.reshape(-1).to(torch.int64)
    S = flat.numel()
    order = torch.argsort(flat, stable=True).to(torch.int32)
    counts = torch.bincount(flat, minlength=E).to(torch.int32)
    off = torch.zeros(E + 1, dtype=torch.int32, device="cuda")
    off[1:] = torch.cumsum(counts, 0)
    sorted_e = flat[order.long()]
    newg = torch.ones(S, dtype=torch.bool, device="cuda")
    newg[1:] = sorted_e[1:] != sorted_e[:-1]
    gidx = torch.cumsum(newg, 0) - 1
    goff = torch.full((S + 1,), S, dtype=torch.int64, device="cuda")
    goff.scatter_reduce_(0, gidx, torch.arange(S, dtype=torch.int64, device="cuda"),
                         reduce="amin")
    gids = torch.zeros(S, dtype=torch.int64, device="cuda")
    gids.scatter_(0, gidx, sorted_e)
    return order, off, gids.to(torch.int32), goff.to(torch.int32)


def t(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


ok = True
for T in (1, 6, 8):
    torch.manual_seed(T)
    hs = (torch.randn(T, HID, device="cuda", dtype=torch.float16) / 8).contiguous()
    _, ids = torch.topk(torch.randn(T, E, device="cuda"), TOPK, dim=-1)
    ids = ids.to(torch.int32)
    order, off, gids, goff = routing(ids)
    S = T * TOPK
    y13a = torch.empty(S, N13, dtype=torch.float16, device="cuda")
    y13b = torch.empty(S, N13, dtype=torch.float16, device="cuda")
    mid = (torch.randn(S, INTER, device="cuda", dtype=torch.float16) / 8).contiguous()
    y2a = torch.empty(S, N2, dtype=torch.float16, device="cuda")
    y2b = torch.empty(S, N2, dtype=torch.float16, device="cuda")

    ext.moe_simt(hs, w13_c, w13_s, g13, order, off, TOPK, y13a, False, T)
    ext.moe_qpn(hs, qc13, qs13, g13, order, gids, goff, TOPK, y13b, False, T, 16, 1)
    ext.moe_simt(mid, w2_c, w2_s, g2, order, off, TOPK, y2a, True, T)
    ext.moe_qpn(mid, qc2, qs2, g2, order, gids, goff, TOPK, y2b, True, T, 8, 1)
    e13 = (y13a.float() - y13b.float()).abs().max().item()
    e2 = (y2a.float() - y2b.float()).abs().max().item()
    r13 = y13a.float().abs().max().item()
    good = e13 <= 2e-3 * max(r13, 1e-3) + 1e-3 and e2 <= 2e-3 * max(
        y2a.float().abs().max().item(), 1e-3) + 1e-3
    ok &= good
    ts13 = t(lambda: ext.moe_simt(hs, w13_c, w13_s, g13, order, off, TOPK, y13a, False, T))
    tq13 = t(lambda: ext.moe_qpn(hs, qc13, qs13, g13, order, gids, goff, TOPK, y13b, False, T, 16, 1))
    ts2 = t(lambda: ext.moe_simt(mid, w2_c, w2_s, g2, order, off, TOPK, y2a, True, T))
    tq2 = t(lambda: ext.moe_qpn(mid, qc2, qs2, g2, order, gids, goff, TOPK, y2b, True, T, 8, 1))
    act = len(set(ids.flatten().tolist()))
    print(f"T={T} ({act} aktive Experten, {S} Slots): diff {e13:.1e}/{e2:.1e} "
          f"{'OK' if good else 'FAIL'} | simt {ts13+ts2:.3f} ms -> qpn {tq13+tq2:.3f} ms "
          f"({(ts13+ts2)/(tq13+tq2):.2f}x)  [w13 {ts13:.3f}->{tq13:.3f}, w2 {ts2:.3f}->{tq2:.3f}]")
print("ALL OK" if ok else "FAILURES")
