"""QPN8 MT=2: correctness and speed vs the chunked path.

MT=2 issues two m8n8k4 row-tiles against one weight stream, so M=9..16 costs
one weight pass instead of chunking's two. Validated against the same
fp8->fp16 reference the native kernel uses, then swept over (split, nacc) and
compared with chunked at the M values the k<=15 verify band actually runs.
"""
import os
import sys

import torch

# Portable defaults: derive from this file's location, never a box path.
# _REPO is the checkout root, so a clone runs without editing paths.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

# qpn8_test.py lives in scripts/, not alongside this file.
sys.path.insert(0, os.path.join(_REPO, "scripts"))
from qpn8_test import ext, qpn8_prepack            # noqa: E402

SHAPES = [(4096, 5120, 48, "gdn in_proj_qkvz"),
          (5120, 1536, 64, "out_proj"),
          (3584, 5120, 16, "attn qkv")]
NATIVE_CFG = {(4096, 5120): (16, 2), (5120, 1536): (8, 2), (3584, 5120): (16, 2)}
SCALE = 0.000972203
dev = torch.device("cuda:0")


def bench(fn, iters=300):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


def operands(n, k):
    torch.manual_seed(0)
    w8 = torch.randint(0, 256, (n, k), dtype=torch.uint8, device=dev)
    w8[w8 == 0x7F] = 0x7E
    w8[w8 == 0xFF] = 0xFE
    ref_w = (w8.view(torch.float8_e4m3fn).float() * SCALE).half()
    packed = qpn8_prepack(w8)
    ts = torch.full((n // 32,), SCALE * 256.0, dtype=torch.float32, device=dev)
    return ref_w, packed, ts


print("=" * 78)
print("1. CORRECTNESS  (vs fp8->fp16 reference; native path scores 2.75e-4)")
print("=" * 78)
allok = True
for (n, k, layers, name) in SHAPES:
    ref_w, packed, ts = operands(n, k)
    sp, na = NATIVE_CFG[(n, k)]
    print(f"\n  {name}  N={n} K={k}  split={sp} nacc={na}")
    for m in (1, 4, 8, 9, 12, 15, 16):
        x = torch.randn(m, k, device=dev, dtype=torch.half) * 0.1
        y_ref = (x.float() @ ref_w.float().t()).half()
        y = ext.gemm_qpn8_mt2(x, packed, ts, n, sp, na)
        err = (y.float() - y_ref.float()).norm() / y_ref.float().norm()
        ok = err < 3e-3 and tuple(y.shape) == (m, n)
        allok &= ok
        print(f"    M={m:3d}  rel_err={err:.2e}  {'OK' if ok else 'FAIL'}")
print("\n  ALL OK" if allok else "\n  FAILURES PRESENT")
_correctness_ok = allok        # exit status is decided at the end

print("\n" + "=" * 78)
print("2. GEOMETRY SWEEP at M=16 (us/call, graph replay)")
print("=" * 78)
best = {}
for (n, k, layers, name) in SHAPES:
    _, packed, ts = operands(n, k)
    x = torch.randn(16, k, device=dev, dtype=torch.half) * 0.1
    print(f"\n  {name}")
    print(f"{'split':>7}{'nacc':>6}{'mt2':>9}{'mt2 fast':>10}")
    for sp in (4, 8, 16):
        if (k // 16) % sp:
            continue
        for na in (1, 2):
            t = bench(lambda sp=sp, na=na: ext.gemm_qpn8_mt2(x, packed, ts, n, sp, na))
            tf = bench(lambda sp=sp, na=na: ext.gemm_qpn8_mt2(x, packed, ts, n, sp, na + 2))
            for label, v in (("", t), ("f", tf)):
                cur = best.get((n, k))
                if cur is None or v < cur[0]:
                    best[(n, k)] = (v, sp, na + (2 if label else 0))
            print(f"{sp:>7}{na:>6}{t:9.2f}{tf:10.2f}")

print("\n" + "=" * 78)
print("3. MT=2 vs CHUNKED at M=16  (the k<=15 verify band)")
print("=" * 78)
print(f"{'shape':>22}{'native M=8':>12}{'chunked M=16':>14}{'MT2 M=16':>10}"
      f"{'MT2 cfg':>10}{'x faster':>10}")
tot_ch = tot_mt2 = 0.0
for (n, k, layers, name) in SHAPES:
    _, packed, ts = operands(n, k)
    sp, na = NATIVE_CFG[(n, k)]
    x8 = torch.randn(8, k, device=dev, dtype=torch.half) * 0.1
    x16 = torch.randn(16, k, device=dev, dtype=torch.half) * 0.1
    t8 = bench(lambda: ext.gemm_qpn8(x8, packed, ts, n, sp, na))

    def ch():
        outs = [ext.gemm_qpn8(x16[i:i + 8].contiguous(), packed, ts, n, sp, na)
                for i in range(0, 16, 8)]
        return torch.cat(outs, 0)
    t_ch = bench(ch)
    bv, bsp, bna = best[(n, k)]
    # Validate the config we are about to recommend. A timing sweep alone
    # will happily crown a kernel that returns NaN -- it did once, and the
    # server accepted 98.8% of drafts at every position because the
    # verifier's logits were garbage. Never report a winner unchecked.
    ref_w, _, _ = operands(n, k)
    y = ext.gemm_qpn8_mt2(x16, packed, ts, n, bsp, bna)
    ref16 = (x16.float() @ ref_w.float().t()).half()
    werr = (y.float() - ref16.float()).norm() / ref16.float().norm()
    assert werr < 3e-3, (f"recommended cfg ({bsp},{bna}) for {name} is "
                         f"NUMERICALLY WRONG: rel_err={werr:.2e}")
    tot_ch += t_ch * layers / 1000.0
    tot_mt2 += bv * layers / 1000.0
    print(f"{name:>22}{t8:12.2f}{t_ch:14.2f}{bv:10.2f}"
          f"{f'({bsp},{bna})':>10}{t_ch / bv:10.2f}")

print(f"\n  128 protected layers at M=16 (per rank, per round):")
print(f"    chunked (deployed)   {tot_ch:6.3f} ms")
print(f"    MT=2                 {tot_mt2:6.3f} ms   "
      f"saves {tot_ch - tot_mt2:.3f} ms/round  ({tot_ch / tot_mt2:.2f}x)")

# Correctness decides the exit status; the sweep above is reporting.
if not _correctness_ok:
    print("\nEXIT 1: MT=2 correctness check failed")
    sys.exit(1)
