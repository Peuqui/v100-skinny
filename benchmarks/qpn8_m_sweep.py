"""Does QPN8 win at every M the server actually uses?

QPN8 was only ever benchmarked at M=8. Production dispatches three ways:

    M <= 8    ext.gemm_qpn8                      (native)
    M 9..16   ceil(M/8) native calls + cat       (chunked; a 2nd weight pass)
    M > 16    unpack -> fp8->fp16 -> F.linear    (reconstruct; prefill)

Those boundaries have never been measured, so this sweeps M and prices every
dispatch that COULD serve each M, on the three protected per-rank shapes,
under CUDA-graph replay (production decode is graphed).

NOTE ON "SIMT": there is no FP8 SIMT kernel -- skinny_fp8_qpn8 is the only
FP8 kernel in the extension, and gemm_simt / gemm_qpn_simt are NVFP4. The
SIMT and QPN2 columns are therefore CROSS-FORMAT references: they price the
same layer as it exists in the all-NVFP4 arm, not an FP8 alternative. The
FP8-vs-FP8 question is native vs chunked vs reconstruct.

Usage: python benchmarks/qpn8_m_sweep.py [--iters 200]
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Portable defaults: derive from this file's location, never a box path.
# _REPO is the checkout root, so a clone runs without editing paths.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")

SHAPES = [
    (5120, 4096, 48, "gdn in_proj_qkvz"),
    (1536, 5120, 64, "out_proj (gdn+attn)"),
    (5120, 3584, 16, "attn qkv"),
]
QPN8_CFG = {(5120, 4096): (16, 2), (1536, 5120): (8, 2), (5120, 3584): (16, 2)}
QPN2_CFG = {(5120, 4096): (16, 2), (1536, 5120): (16, 2), (5120, 3584): (16, 2)}
MS = (1, 2, 4, 8, 12, 16, 24, 32, 64)


def bench(fn, iters, graph=True):
    try:
        fn()
    except Exception as e:
        return None, type(e).__name__
    torch.cuda.synchronize()
    run = fn
    if graph:
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            torch.cuda.synchronize()
            run = g.replay
        except Exception:
            run = fn                      # not capturable; time eager
    for _ in range(10):
        run()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        run()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    ext = load(name="skinny_nvfp4_v11",
               sources=[os.environ.get("VLLM_SKINNY_NVFP4_SRC",
                                       os.path.join(_REPO, "kernels", "skinny_kernels.cu"))],
               extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo",
                                  "-gencode=arch=compute_70,code=sm_70"],
               verbose=False)
    dev = torch.device("cuda:0")
    print("all times us/call, CUDA-graph replay; '-' = shape not supported")

    for (k, n, layers, name) in SHAPES:
        sp8, na8 = QPN8_CFG[(k, n)]
        sp2, na2 = QPN2_CFG[(k, n)]
        c8 = torch.randint(0, 254, (n * k,), dtype=torch.uint8, device=dev)
        ts = torch.full((n // 32,), 0.01, dtype=torch.float32, device=dev)
        c4 = torch.randint(0, 256, (n * k // 2,), dtype=torch.uint8, device=dev)
        s4 = torch.randint(1, 200, (n * k // 16,), dtype=torch.uint8, device=dev)
        # reconstruct operand: a plain (N,K) fp8 tensor, as the prefill path
        # materialises it before the fp16 GEMM
        w8 = torch.randint(0, 254, (n, k), dtype=torch.uint8,
                           device=dev).view(torch.float8_e4m3fn)
        colscale = torch.full((n, 1), 0.01, dtype=torch.half, device=dev)

        print(f"\n=== {name}  K={k} N={n}  (x{layers} layers/round) ===")
        print(f"{'M':>4}{'qpn8':>10}{'chunked':>10}{'reconstruct':>13}"
              f"{'| qpn2':>10}{'simt':>10}   winner (FP8 paths only)")
        for m in MS:
            x = torch.randn(m, k, device=dev, dtype=torch.half) * 0.1

            nat, _ = (bench(lambda: ext.gemm_qpn8(x, c8, ts, n, sp8, na8),
                            args.iters) if m <= 8 else (None, "M>8"))

            def chunked(m=m, x=x):
                outs = [ext.gemm_qpn8(x[i:i + 8].contiguous(), c8, ts, n,
                                      sp8, na8) for i in range(0, m, 8)]
                return torch.cat(outs, 0) if len(outs) > 1 else outs[0]
            chk, _ = bench(chunked, args.iters)

            def recon(x=x):
                return F.linear(x, w8.to(x.dtype) * colscale)
            rec, _ = bench(recon, args.iters)

            q2, _ = (bench(lambda: ext.gemm_qpn2(x, c4, s4, 1.0, n, sp2, na2),
                           args.iters) if m <= 8 else (None, "M>8"))
            st, _ = (bench(lambda: ext.gemm_simt(x, c4, s4, 1.0, n),
                           args.iters) if m <= 8 else (None, "M>8"))

            cands = {"qpn8": nat, "chunked": chk, "reconstruct": rec}
            live = {kk: v for kk, v in cands.items() if v is not None}
            win = min(live, key=live.get) if live else "-"
            f = lambda v: f"{v:.2f}" if v is not None else "-"
            print(f"{m:>4}{f(nat):>10}{f(chk):>10}{f(rec):>13}"
                  f"{f(q2):>10}{f(st):>10}   {win}")

        print(f"  deployed boundary: native<=8, chunked 9..16, reconstruct>16")

    mt2_sizing(ext, dev, args.iters)
    prefill_crossover(ext, dev, max(args.iters // 4, 25))


def prefill_crossover(ext, dev, iters):
    """Where does reconstruct actually start beating chunking?

    Production switches at M>16, but the sweep shows chunking still winning
    at M=64 by 1.7x. Reconstruct pays a fixed dequant of the whole weight
    matrix and then one dense fp16 GEMM, so it must win eventually -- this
    finds where.
    """
    print("\n" + "=" * 74)
    print("PREFILL CROSSOVER: chunked vs reconstruct at large M")
    print("=" * 74)
    for (k, n, layers, name) in SHAPES:
        sp8, na8 = QPN8_CFG[(k, n)]
        c8 = torch.randint(0, 254, (n * k,), dtype=torch.uint8, device=dev)
        ts = torch.full((n // 32,), 0.01, dtype=torch.float32, device=dev)
        w8 = torch.randint(0, 254, (n, k), dtype=torch.uint8,
                           device=dev).view(torch.float8_e4m3fn)
        cs = torch.full((n, 1), 0.01, dtype=torch.half, device=dev)
        print(f"\n  {name}  K={k} N={n}")
        print(f"{'M':>6}{'chunked':>11}{'reconstruct':>13}{'winner':>13}")
        for m in (16, 32, 64, 96, 128, 192, 256, 384):
            x = torch.randn(m, k, device=dev, dtype=torch.half) * 0.1

            def ch(x=x, m=m):
                outs = [ext.gemm_qpn8(x[i:i + 8].contiguous(), c8, ts, n,
                                      sp8, na8) for i in range(0, m, 8)]
                return torch.cat(outs, 0)
            t_ch, _ = bench(ch, iters)
            t_rc, _ = bench(lambda x=x: F.linear(x, w8.to(x.dtype) * cs), iters)
            win = "chunked" if t_ch < t_rc else "reconstruct"
            print(f"{m:>6}{t_ch:11.2f}{t_rc:13.2f}{win:>13}")


def mt2_sizing(ext, dev, iters):
    """What would an MT=2 kernel (two m8n8k4 tiles, ONE weight stream) buy?

    m8n8k4 gives 8 rows per tile, so M=16 needs two tiles. Chunking issues
    them as two separate calls and therefore streams the weights TWICE --
    that, not the tile quantization, is what makes the k<=15 band cost
    298 GB/s against the k<=7 band's 431.

    An MT=2 kernel would stream the weights once and issue two tiles, so its
    cost is T(M=8) plus whatever a second tile costs when the weights are
    already in flight: 8 more rows of activation traffic, 8 more rows of
    epilogue writes, and one more MMA issue.

    Two independent estimates of that increment:

      (a) the per-row slope of T(M) across M=1..8, where the weight stream is
          already constant and only A/C traffic varies;
      (b) a direct measurement on a shape whose weights fit in the V100's
          6 MB L2, where the second chunked pass re-reads from cache rather
          than DRAM -- so chunked-vs-native there isolates the second-tile
          cost with the weight re-read removed.
    """
    L2_BYTES = 6 << 20
    print("\n" + "=" * 74)
    print("MT=2 SIZING  (two tiles, one weight stream) -- PROJECTION, not a kernel")
    print("=" * 74)

    print("\n(b) second-tile cost with weights L2-resident.")
    print("    The probe shape must be L2-resident AND realistically wide --")
    print("    a narrow shape has too few CTAs and measures launch latency,")
    print("    not the tile cost. Real shapes run 112-160 CTAs (grid = N/32).")
    print(f"{'K':>6}{'N':>6}{'MB':>7}{'CTAs':>6}{'M=8':>9}{'M=16 chunk':>12}"
          f"{'increment':>11}{'ratio':>8}")
    incs = []
    for (k, n) in ((1024, 4096), (1280, 4096), (1024, 5120)):
        mb = n * k / 2**20
        if mb > L2_BYTES / 2**20:
            continue
        c8 = torch.randint(0, 254, (n * k,), dtype=torch.uint8, device=dev)
        ts = torch.full((n // 32,), 0.01, dtype=torch.float32, device=dev)
        x8 = torch.randn(8, k, device=dev, dtype=torch.half) * 0.1
        x16 = torch.randn(16, k, device=dev, dtype=torch.half) * 0.1
        t8, _ = bench(lambda: ext.gemm_qpn8(x8, c8, ts, n, 8, 2), iters)

        def ch(x=x16):
            outs = [ext.gemm_qpn8(x[i:i + 8].contiguous(), c8, ts, n, 8, 2)
                    for i in range(0, 16, 8)]
            return torch.cat(outs, 0)
        t16, _ = bench(ch, iters)
        if t8 and t16:
            incs.append((t16 - t8) / t8)
            print(f"{k:>6}{n:>6}{mb:7.2f}{n // 32:6d}{t8:9.2f}{t16:12.2f}"
                  f"{t16 - t8:11.2f}{t16 / t8:8.3f}")
    inc = sum(incs) / len(incs) if incs else None
    if inc is not None:
        print(f"  mean second-tile increment (weights cached) = "
              f"+{inc * 100:.1f}% of T(M=8)")
        print("  If the kernel were DRAM-bound this would be near 0% (the "
              "second tile\n  re-reads from L2). Near +100% instead means the "
              "cost is the tile itself,\n  not the weight traffic -- and MT=2 "
              "would buy nothing.")

    print("\n(a-bis) effective bandwidth at M=8 on the REAL shapes "
          "(copy ceiling ~825 GB/s):")
    for (k, n, layers, name) in SHAPES:
        sp8, na8 = QPN8_CFG[(k, n)]
        c8 = torch.randint(0, 254, (n * k,), dtype=torch.uint8, device=dev)
        ts = torch.full((n // 32,), 0.01, dtype=torch.float32, device=dev)
        x8 = torch.randn(8, k, device=dev, dtype=torch.half) * 0.1
        t8, _ = bench(lambda: ext.gemm_qpn8(x8, c8, ts, n, sp8, na8), iters)
        gbs = (n * k) / (t8 * 1e-6) / 1e9
        print(f"    {name:>22}  {t8:6.2f} us  {gbs:6.1f} GB/s  "
              f"{100 * gbs / 825:5.1f}% of ceiling  ({n // 32} CTAs)")

    print("\n(a) per-row slope from M=1..8 on the real shapes, and the projection:")
    print(f"{'shape':>22}{'T(M=1)':>9}{'T(M=8)':>9}{'us/row':>8}"
          f"{'chunk16':>9}{'MT2 est':>9}{'saving':>9}{'x faster':>9}")
    tot_chunk = tot_mt2 = 0.0
    for (k, n, layers, name) in SHAPES:
        sp8, na8 = QPN8_CFG[(k, n)]
        c8 = torch.randint(0, 254, (n * k,), dtype=torch.uint8, device=dev)
        ts = torch.full((n // 32,), 0.01, dtype=torch.float32, device=dev)
        x1 = torch.randn(1, k, device=dev, dtype=torch.half) * 0.1
        x8 = torch.randn(8, k, device=dev, dtype=torch.half) * 0.1
        x16 = torch.randn(16, k, device=dev, dtype=torch.half) * 0.1
        t1, _ = bench(lambda: ext.gemm_qpn8(x1, c8, ts, n, sp8, na8), iters)
        t8, _ = bench(lambda: ext.gemm_qpn8(x8, c8, ts, n, sp8, na8), iters)

        def ch(x=x16):
            outs = [ext.gemm_qpn8(x[i:i + 8].contiguous(), c8, ts, n, sp8,
                                  na8) for i in range(0, 16, 8)]
            return torch.cat(outs, 0)
        t16, _ = bench(ch, iters)
        slope = (t8 - t1) / 7.0
        est_a = t8 + 8 * slope
        est_b = t8 * (1 + inc) if inc is not None else est_a
        est = max(est_a, est_b)          # the more conservative of the two
        tot_chunk += t16 * layers / 1000.0
        tot_mt2 += est * layers / 1000.0
        print(f"{name:>22}{t1:9.2f}{t8:9.2f}{slope:8.3f}"
              f"{t16:9.2f}{est:9.2f}{t16 - est:9.2f}{t16 / est:9.2f}")

    print(f"\n  128 protected layers at M=16 (the k=15 verify):")
    print(f"    chunked (deployed)   {tot_chunk:6.3f} ms")
    print(f"    MT=2 (projected)     {tot_mt2:6.3f} ms    "
          f"saves {tot_chunk - tot_mt2:.3f} ms/round")
    print("  Projection from measured slopes, NOT a kernel measurement. The "
          "MT=2\n  estimate is a FLOOR-ish figure: it assumes the second tile "
          "adds only\n  activation/epilogue traffic and one MMA issue, and no "
          "new register or\n  occupancy pressure -- which is exactly the risk "
          "an implementation carries.")


if __name__ == "__main__":
    main()
