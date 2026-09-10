"""qpn2 with and without the activation block-pack, on the serving shapes.

Builds two kernel sources side by side and runs both through identical
inputs: bit-identity first, then CUDA-graph replay timing. Graph replay is
the regime that matters -- decode runs captured, so an extra kernel costs
graph-replay overhead and not the eager launch path, and an eager harness
overstates the pack by a factor of three.

The block-pack is what closed the Turing DFlash2 gap on 2026-09-09
(69.22 -> 72.72 tok/s, same text SHA). Background and the measurements it
rests on: STAND.md point 8.

Acceptance is torch.equal, not a tolerance: the change moves bytes, it does
not change arithmetic, so any difference at all is a defect.

Two things this harness gets right that earlier attempts got wrong:

  * Data is tame. Random code bytes with random e4m3 scales and gscale=1.0
    run gscale*16384 into fp16 overflow, and the comparison then reads back
    NaN -- which says nothing about either kernel. Scales are exactly 1.0
    (e4m3 0x38) and gscale is 1/16384, so the dequantised weights are the
    nvfp4 nibble values themselves, and the reference is checked finite.
  * It reports every M, including the ones below the pack threshold where
    both sides run the SAME code. Those cells must read exactly 1.00x. When
    they do not, the change is not as neutral as it looks -- that is how the
    runtime-stride version was caught costing the V100 1-5%.

Usage:
    CUDA_VISIBLE_DEVICES=<one card> PROBE_ARCH=<70|75> \\
        python benchmarks/qpn2_pack_ab.py [--ref <rev-or-path>]

--ref takes a git revision (default HEAD~1) or a path to a .cu file.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
NEW = os.path.join(_REPO, "kernels", "skinny_kernels.cu")
ARCH = os.environ.get("PROBE_ARCH", "75")
dev = "cuda"
GSCALE = 1.0 / 16384.0
SCALE_BYTE = 0x38  # e4m3 1.0
LM_HEAD = (5120, 62080)

from vllm.model_executor.kernels.linear.nvfp4 import marlin as shim  # noqa: E402


def resolve_ref(ref):
    if os.path.exists(ref):
        return ref
    blob = subprocess.run(
        ["git", "-C", _REPO, "show", f"{ref}:kernels/skinny_kernels.cu"],
        capture_output=True, check=True).stdout
    fd, path = tempfile.mkstemp(suffix=".cu", prefix="skinny_ref_")
    with os.fdopen(fd, "wb") as fh:
        fh.write(blob)
    return path


def build(name, src):
    return load(name=name, sources=[src],
                extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo",
                                   f"-gencode=arch=compute_{ARCH},code=sm_{ARCH}"],
                verbose=False)


def read_ceiling():
    n = 512 * 1024 * 1024 // 2
    a = torch.empty(n, dtype=torch.float16, device=dev).normal_()
    nbytes = a.numel() * a.element_size()
    for _ in range(5):
        torch.sum(a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        torch.sum(a)
    torch.cuda.synchronize()
    gbs = nbytes / ((time.perf_counter() - t0) / 30) / 1e9
    del a
    torch.cuda.empty_cache()
    return gbs


def graph_bench(fn, it=2000):
    fn()
    torch.cuda.synchronize()
    gph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph):
        fn()
    torch.cuda.synchronize()
    for _ in range(20):
        gph.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it):
        gph.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD~1",
                    help="git revision or .cu path to compare against")
    args = ap.parse_args()

    props = torch.cuda.get_device_properties(0)
    print(f"# device: {props.name}  built sm_{ARCH}  CUDA-graph replay")
    print(f"# ref = {args.ref}   new = working tree")
    ceil = read_ceiling()
    print(f"# read ceiling: {ceil:.0f} GB/s")
    ref = build(f"qpn2ab_ref_sm{ARCH}", resolve_ref(args.ref))
    new = build(f"qpn2ab_new_sm{ARCH}", NEW)

    g = torch.Generator(device="cpu").manual_seed(0)
    bad = 0
    tot_r = tot_n = 0.0
    print("K,N,M,bitequal,ref_GBs,new_GBs,speedup,new_pct")
    for k, n in sorted(shim._QPN2_TABLE.keys()):
        codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8,
                              generator=g).to(dev)
        sb = torch.full((n, k // 16), SCALE_BYTE, dtype=torch.uint8, device=dev)
        qc, qs = shim._qpn_prepack(codes, sb)
        cfg = shim._qpn2_cfg(k, n)
        if qc is None or cfg is None:
            continue
        gb = n * (k // 2) + n * (k // 16)
        for M in range(1, 9):
            x = (torch.randint(-8, 9, (M, k), generator=g).half() / 8.0).to(dev)
            yr = ref.gemm_qpn2(x, qc, qs, GSCALE, n, cfg[0], cfg[1])
            yn = new.gemm_qpn2(x, qc, qs, GSCALE, n, cfg[0], cfg[1])
            eq = torch.equal(yr, yn)
            finite = bool(torch.isfinite(yr).all())
            if not eq or not finite:
                bad += 1
            tr = graph_bench(lambda: ref.gemm_qpn2(x, qc, qs, GSCALE, n,
                                                   cfg[0], cfg[1]))
            tn = graph_bench(lambda: new.gemm_qpn2(x, qc, qs, GSCALE, n,
                                                   cfg[0], cfg[1]))
            if M == 8 and (k, n) != LM_HEAD:
                tot_r += tr
                tot_n += tn
            gr, gn = gb / (tr * 1e-3) / 1e9, gb / (tn * 1e-3) / 1e9
            print(f"{k},{n},{M},{'YES' if eq else 'NO'}"
                  f"{'' if finite else '/NONFINITE'},{gr:.1f},{gn:.1f},"
                  f"{gn / gr:.2f}x,{gn / ceil * 100:.0f}%", flush=True)
        # Rebind rather than `del`: the lambdas above close over qc/qs, and
        # deleting the names drops them for the whole scope. lm_head alone is
        # ~179 MB packed, so the release itself is not optional.
        codes = sb = qc = qs = None
        torch.cuda.empty_cache()
    print(f"# trunk aggregate at M=8: {tot_r * 1e3:.1f}us -> {tot_n * 1e3:.1f}us"
          f" = {tot_r / tot_n:.2f}x")
    print(f"# mismatches: {bad}")
    sys.exit(1 if bad else 0)


main()
