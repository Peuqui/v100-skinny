"""Drive kernels/mma_probe.cu: verify fragment layouts, then price both shapes.

Replaces the mma8_probe.cu the kernel comments refer to, which was lost.
Answers two questions the 09.09. Turing work had to settle:

  1. What IS the m16n8k8 fragment layout on this hardware? Verified against
     a CPU reference rather than taken from the PTX manual -- a wrong map
     cannot pass. The production m8n8k4 map is checked the same way.
  2. Is Voltas m8n8k4 slower than Turings m16n8k8 on a Turing card? At
     equal FLOP count, registers only, no memory in the inner loop.

Measured 2026-09-09 on the RTX 8000: 45.2 vs 45.0 TFLOPS at saturation --
indistinguishable, and the m8n8k4 arm is the handicapped one (one dependent
accumulator chain against two independent). There is no MMA shape penalty
on Turing. The absolute numbers are latency-bound, not a roofline; only the
comparison under identical conditions is meant to carry weight.

Usage: CUDA_VISIBLE_DEVICES=<one card> PROBE_ARCH=75 python benchmarks/mma_probe.py

Correctness comes first and gates the timing: a layout that fails the CPU
reference makes every number after it meaningless.
"""
import os
import time

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(HERE)
ARCH = os.environ.get("PROBE_ARCH", "75")

ext = load(name=f"mma_probe_sm{ARCH}",
           sources=[os.path.join(_REPO, "kernels", "mma_probe.cu")],
           extra_cuda_cflags=["-O3", "-lineinfo",
                              f"-gencode=arch=compute_{ARCH},code=sm_{ARCH}"],
           verbose=False)

dev = "cuda"
cap = torch.cuda.get_device_capability()
print(f"# device: {torch.cuda.get_device_name(0)}  cap sm_{cap[0]}{cap[1]}  "
      f"built for sm_{ARCH}")

g = torch.Generator(device="cpu").manual_seed(7)


def check(name, D, ref):
    err = (D.cpu().float() - ref).abs().max().item()
    ok = err < 1e-2
    print(f"# LAYOUT {name}: {'OK' if ok else 'WRONG'}  max|err|={err:.3e}")
    return ok


# ---- m16n8k8: D[16x8] = A[16x8] @ B[8x8] ---------------------------------
A = (torch.randint(-4, 5, (16, 8), generator=g).half())
Bt = (torch.randint(-4, 5, (8, 8), generator=g).half())   # [n][k], col-major
ref = A.float() @ Bt.float().t()
ok8 = check("m16n8k8", ext.run_m16n8k8(A.to(dev).contiguous(),
                                       Bt.to(dev).contiguous()), ref)

# ---- m8n8k4: four QP tiles, D[8x32] = A[8x4] @ B[32x4]^T ------------------
A4 = (torch.randint(-4, 5, (8, 4), generator=g).half())
B4 = (torch.randint(-4, 5, (32, 4), generator=g).half())  # [n][k]
ref4 = A4.float() @ B4.float().t()
ok4 = check("m8n8k4", ext.run_m8n8k4(A4.to(dev).contiguous(),
                                     B4.to(dev).contiguous()), ref4)

if not (ok8 and ok4):
    raise SystemExit("# layout check failed -- timings withheld")

# ---- issue rate, equal FLOP per iteration --------------------------------
# 2048 FLOP per iteration per warp in both arms (one m8n8k4 = two m16n8k8).
SMS = torch.cuda.get_device_properties(0).multi_processor_count
ITERS = 20000
print(f"# {SMS} SMs, {ITERS} iterations/warp, 2048 FLOP/iteration/warp")
print("shape,blocks,warps,ms,TFLOPs")
for blocks, warps in ((SMS, 4), (SMS * 2, 4), (SMS * 4, 8)):
    for shape, label in ((4, "m8n8k4"), (8, "m16n8k8")):
        ext.bench_issue(shape, blocks, warps, 100)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ext.bench_issue(shape, blocks, warps, ITERS)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        flop = blocks * warps * ITERS * 2048.0
        print(f"{label},{blocks},{warps},{dt * 1e3:.2f},{flop / dt / 1e12:.1f}",
              flush=True)
print("MMA_PROBE_DONE")
