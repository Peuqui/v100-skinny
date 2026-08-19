"""QPN8 correctness: real-shape FP8 GEMM vs an fp8->fp16 reference.

Prepack is QPN2's permutation at byte granularity -- same lane->column map,
same korder (which cancels the decoder's (i, i+4) interleave). No arithmetic
touches a stored code.
"""
import os
import sys

import torch
from torch.utils.cpp_extension import load

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")

# Portable defaults: derive from this file's location, never a box path.
# _REPO is the checkout root, so a clone runs without editing paths.
# SKINNY_KERNELS_SRC overrides for out-of-tree kernel sources.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SRC = os.environ.get("SKINNY_KERNELS_SRC",
                      os.path.join(_REPO, "kernels", "skinny_kernels.cu"))

ext = load(name="skinny_nvfp4_v11",
           sources=[_SRC],
           extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo",
                              "-gencode=arch=compute_70,code=sm_70"],
           verbose=False)
print("extension loaded; has gemm_qpn8:", hasattr(ext, "gemm_qpn8"))

KORDER = [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15]


def qpn8_prepack(w8: torch.Tensor) -> torch.Tensor:
    """(N,K) uint8 -> [tile][group][lane][16B] fragment order."""
    n, k = w8.shape
    assert n % 32 == 0 and k % 16 == 0
    dev = w8.device
    tiles, groups = n // 32, k // 16
    lane = torch.arange(32, device=dev)
    col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0).long() * 4
    korder = torch.tensor(KORDER, device=dev)
    g = torch.arange(groups, device=dev)
    kidx = g.view(groups, 1) * 16 + korder.view(1, 16)
    out = torch.empty(tiles, groups, 32, 16, dtype=torch.uint8, device=dev)
    chunk = max(1, 36864 // max(groups, 1))
    for t0 in range(0, tiles, chunk):
        t1 = min(t0 + chunk, tiles)
        tt = t1 - t0
        ncol = (torch.arange(t0, t1, device=dev).view(tt, 1) * 32
                + col.view(1, 32))
        out[t0:t1] = w8[ncol.view(tt, 1, 32, 1).expand(tt, groups, 32, 16),
                        kidx.view(1, groups, 1, 16).expand(tt, groups, 32, 16)]
    return out.view(-1).contiguous()


def check(n, k, m, splitk, nacc, scale=0.000972203):
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    # random E4M3 codes, avoiding the two NaN encodings
    w8 = torch.randint(0, 256, (n, k), dtype=torch.uint8, device=dev)
    w8[w8 == 0x7F] = 0x7E
    w8[w8 == 0xFF] = 0xFE
    ref_w = (w8.view(torch.float8_e4m3fn).float() * scale).half()
    x = (torch.randn(m, k, device=dev, dtype=torch.half) * 0.1)
    y_ref = (x.float() @ ref_w.float().t()).half()

    packed = qpn8_prepack(w8)
    tscale = torch.full((n // 32,), scale * 256.0, dtype=torch.float32,
                        device=dev)
    y = ext.gemm_qpn8(x, packed, tscale, n, splitk, nacc)
    err = (y.float() - y_ref.float()).norm() / y_ref.float().norm()
    ok = err < 3e-3
    print("N=%5d K=%5d M=%d split=%2d nacc=%d  rel_err=%.2e  %s"
          % (n, k, m, splitk, nacc, err, "OK" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    shapes = [(4096, 5120), (5120, 1536), (3584, 5120), (5120, 6144)]
    allok = True
    for (n, k) in shapes:
        for splitk in (8, 16):
            allok &= check(n, k, 8, splitk, 1)
    for m in (1, 2, 4, 7, 8):
        allok &= check(4096, 5120, m, 16, 2)
    print("ALL OK" if allok else "FAILURES PRESENT")
    sys.exit(0 if allok else 1)
