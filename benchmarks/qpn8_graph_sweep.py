"""QPN8 geometry sweep under CUDA-graph replay.

QPN8's (split-K, nacc) table was chosen from an EAGER sweep. QPN2's own tuning
is known to diverge by up to 27% between eager and graph mode, and production
decode runs entirely under graph replay, so the deployed QPN8 table has never
been validated in the mode it actually runs in.

Sweeps the three protected shapes at M=8 in both modes and reports the
per-round cost of all 128 protected layers under the deployed table versus the
graph-mode optimum. QPN2 on the same shapes is included as the floor.

Usage: python benchmarks/qpn8_graph_sweep.py [--iters 300]
"""
import argparse
import os

import torch
from torch.utils.cpp_extension import load

# Portable defaults: derive from this file's location, never a box path.
# _REPO is the checkout root, so a clone runs without editing paths.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")

# (K, N, layers/round, name) per rank, TP=4
SHAPES = [
    (5120, 4096, 48, "gdn in_proj_qkvz"),
    (1536, 5120, 64, "out_proj (gdn+attn)"),
    (5120, 3584, 16, "attn qkv"),
]
# deployed QPN8 table, keyed (N, K) as in modelopt._SM70_QPN8_TABLE
DEPLOYED = {(4096, 5120): (16, 2), (5120, 1536): (8, 2), (3584, 5120): (16, 2)}
QPN2_CFG = {(5120, 4096): (16, 2), (1536, 5120): (16, 2), (5120, 3584): (16, 2)}
SPLITS = (8, 16, 32)
NACCS = (1, 2)
M = 8


def bench(fn, iters, graph):
    fn()
    torch.cuda.synchronize()
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        run = g.replay
    else:
        run = fn
    for _ in range(20):
        run()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        run()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0      # microseconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()
    ext = load(name="skinny_nvfp4_v11",
               sources=[os.environ.get("VLLM_SKINNY_NVFP4_SRC",
                                       os.path.join(_REPO, "kernels", "skinny_kernels.cu"))],
               extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo",
                                  "-gencode=arch=compute_70,code=sm_70"],
               verbose=False)
    dev = torch.device("cuda:0")

    dep_tot = best_tot = qpn2_tot = 0.0
    changes = []
    for (k, n, layers, name) in SHAPES:
        x = torch.randn(M, k, device=dev, dtype=torch.half) * 0.1
        c8 = torch.randint(0, 254, (n * k,), dtype=torch.uint8, device=dev)
        ts = torch.full((n // 32,), 0.01, dtype=torch.float32, device=dev)
        c4 = torch.randint(0, 256, (n * k // 2,), dtype=torch.uint8, device=dev)
        s4 = torch.randint(1, 200, (n * k // 16,), dtype=torch.uint8, device=dev)

        print(f"\n=== {name}  K={k} N={n}  x{layers} layers/round ===")
        print(f"{'split':>6}{'nacc':>6}{'eager us':>10}{'graph us':>10}"
              f"{'graph GB/s':>12}   note")
        best = None
        dep_cfg = DEPLOYED[(n, k)]
        for sp in SPLITS:
            if (k // 16) % sp:
                continue
            for na in NACCS:
                f = (lambda sp=sp, na=na:
                     ext.gemm_qpn8(x, c8, ts, n, sp, na))
                eg = bench(f, args.iters, False)
                gr = bench(f, args.iters, True)
                gbs = (n * k) / (gr * 1e-6) / 1e9
                tag = " <- deployed" if (sp, na) == dep_cfg else ""
                if best is None or gr < best[2]:
                    best = (sp, na, gr)
                print(f"{sp:6d}{na:6d}{eg:10.2f}{gr:10.2f}{gbs:12.1f}{tag}")

        dep_us = bench(lambda: ext.gemm_qpn8(x, c8, ts, n, *dep_cfg),
                       args.iters, True)
        sp2, na2 = QPN2_CFG[(k, n)]
        q2 = bench(lambda: ext.gemm_qpn2(x, c4, s4, 1.0, n, sp2, na2),
                   args.iters, True)
        dep_tot += dep_us * layers / 1000.0
        best_tot += best[2] * layers / 1000.0
        qpn2_tot += q2 * layers / 1000.0
        print(f"  deployed {dep_cfg} = {dep_us:.2f} us | graph-best "
              f"({best[0]},{best[1]}) = {best[2]:.2f} us "
              f"({100*(best[2]-dep_us)/dep_us:+.1f}%) | qpn2 = {q2:.2f} us")
        if (best[0], best[1]) != dep_cfg and best[2] < dep_us - 0.05:
            changes.append((name, n, k, dep_cfg, (best[0], best[1]),
                            dep_us, best[2]))

    print(f"\n=== 128 protected layers/round (graph replay, M=8) ===")
    print(f"  QPN2 (arm A)              {qpn2_tot:6.3f} ms")
    print(f"  QPN8 deployed table       {dep_tot:6.3f} ms   "
          f"(+{dep_tot - qpn2_tot:.3f} vs QPN2)")
    print(f"  QPN8 graph-mode optimum   {best_tot:6.3f} ms   "
          f"(+{best_tot - qpn2_tot:.3f} vs QPN2, "
          f"{dep_tot - best_tot:.3f} ms saved)")
    if changes:
        print("\n  table changes worth making (key is (N, K)):")
        for nm, n, k, old, new, ou, bu in changes:
            print(f"    ({n}, {k}): {old} -> {new}   {ou:.2f} -> {bu:.2f} us   ({nm})")
    else:
        print("\n  deployed table is already the graph-mode optimum on all three shapes")


if __name__ == "__main__":
    main()
