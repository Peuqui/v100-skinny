"""QPN8-BLK correctness: block-scaled FP8 GEMM vs an fp8->fp16 reference.

Same prepack as qpn8_test.py (QPN2's byte permutation). The scale is the
checkpoint-style [ceil(N/bn), ceil(K/bk)] fp32 raster; every block gets a
DISTINCT scale so any mis-indexed block lookup fails loudly instead of
averaging out.
"""
import os
import sys

import torch
from torch.utils.cpp_extension import load

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SRC = os.environ.get("SKINNY_KERNELS_SRC",
                      os.path.join(_REPO, "kernels", "skinny_kernels.cu"))

ext = load(name="skinny_nvfp4_v11",
           sources=[_SRC],
           extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo",
                              "-gencode=arch=compute_70,code=sm_70"],
           verbose=False)
print("extension loaded; has gemm_qpn8_blk:", hasattr(ext, "gemm_qpn8_blk"),
      "gemm_qpn8_blk_mt2:", hasattr(ext, "gemm_qpn8_blk_mt2"))

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


def make_case(n, k, m, bn, bk, dev, seed=0):
    torch.manual_seed(seed)
    w8 = torch.randint(0, 256, (n, k), dtype=torch.uint8, device=dev)
    w8[w8 == 0x7F] = 0x7E
    w8[w8 == 0xFF] = 0xFE
    nb, kb = (n + bn - 1) // bn, (k + bk - 1) // bk
    # log-uniform in [2^-12, 2^-8), all blocks distinct
    bscale = (2.0 ** (torch.rand(nb, kb, device=dev) * 4.0 - 12.0)).float()
    sc_full = (bscale.repeat_interleave(bn, 0)[:n]
               .repeat_interleave(bk, 1)[:, :k])
    ref_w = (w8.view(torch.float8_e4m3fn).float() * sc_full)
    x = (torch.randn(m, k, device=dev, dtype=torch.half) * 0.1)
    y_ref = (x.float() @ ref_w.t()).half()
    packed = qpn8_prepack(w8)
    return x, packed, bscale, y_ref


def check(n, k, m, bn, bk, splitk, nacc, mt2=False):
    dev = torch.device("cuda:0")
    x, packed, bscale, y_ref = make_case(n, k, m, bn, bk, dev)
    fn = ext.gemm_qpn8_blk_mt2 if mt2 else ext.gemm_qpn8_blk
    y = fn(x, packed, bscale, n, bn, bk, splitk, nacc)
    err = (y.float() - y_ref.float()).norm() / y_ref.float().norm()
    ok = err < 3e-3
    print("N=%5d K=%5d M=%2d bn=%3d bk=%3d split=%2d nacc=%d %-4s rel_err="
          "%.2e  %s" % (n, k, m, bn, bk, splitk, nacc,
                        "mt2" if mt2 else "", err, "OK" if ok else "FAIL"))
    return ok


def bench(n, k, m, splitk, nacc, iters=200):
    """Per-tile QPN8 vs block-scaled QPN8, same shape, same config."""
    dev = torch.device("cuda:0")
    x, packed, bscale, _ = make_case(n, k, m, 128, 128, dev)
    tscale = torch.full((n // 32,), 2.0 ** -10 * 256.0, dtype=torch.float32,
                        device=dev)

    def time_it(fn, *args):
        for _ in range(20):
            fn(*args)
        torch.cuda.synchronize()
        beg = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        beg.record()
        for _ in range(iters):
            fn(*args)
        end.record()
        torch.cuda.synchronize()
        return beg.elapsed_time(end) / iters

    t_ref = time_it(ext.gemm_qpn8, x, packed, tscale, n, splitk, nacc)
    t_blk = time_it(ext.gemm_qpn8_blk, x, packed, bscale, n, 128, 128,
                    splitk, nacc)
    gb = n * k / 1e9
    print("BENCH N=%5d K=%5d M=%d split=%2d nacc=%d  tile %.3f ms (%.0f GB/s)"
          "  blk %.3f ms (%.0f GB/s)  overhead %+.1f%%"
          % (n, k, m, splitk, nacc, t_ref, gb / (t_ref / 1e3),
             t_blk, gb / (t_blk / 1e3), (t_blk / t_ref - 1) * 100))


if __name__ == "__main__":
    shapes = [(4096, 5120), (5120, 1536), (3584, 5120), (5120, 6144)]
    allok = True
    for (n, k) in shapes:
        for splitk in (8, 16):
            allok &= check(n, k, 8, 128, 128, splitk, 1)
    for m in (1, 2, 4, 7, 8):
        allok &= check(4096, 5120, m, 128, 128, 16, 2)
    # fast decoder
    for m in (1, 8):
        allok &= check(4096, 5120, m, 128, 128, 16, 3)
    # MT2 band
    for m in (9, 12, 16):
        allok &= check(4096, 5120, m, 128, 128, 16, 3, mt2=True)
        allok &= check(4096, 5120, m, 128, 128, 8, 2, mt2=True)
    # WMMA prefill band (reads the same packed layout; K % 128 == 0)
    for m in (17, 32, 64, 100, 256, 2048):
        dev = torch.device("cuda:0")
        x, packed, bscale, y_ref = make_case(4096, 5120, m, 128, 128, dev)
        y = ext.gemm_qpn8_blk_wmma(x, packed, bscale, 4096, 128, 128)
        err = (y.float() - y_ref.float()).norm() / y_ref.float().norm()
        ok = err < 3e-3
        allok &= bool(ok)
        print("N= 4096 K= 5120 M=%4d bn=128 bk=128 wmma      rel_err=%.2e  %s"
              % (m, err, "OK" if ok else "FAIL"))
    # wmma on non-128 raster geometry
    x, packed, bscale, y_ref = make_case(4096, 5120, 64, 64, 64,
                                         torch.device("cuda:0"))
    y = ext.gemm_qpn8_blk_wmma(x, packed, bscale, 4096, 64, 64)
    err = (y.float() - y_ref.float()).norm() / y_ref.float().norm()
    allok &= bool(err < 3e-3)
    print("N= 4096 K= 5120 M=  64 bn= 64 bk= 64 wmma      rel_err=%.2e  %s"
          % (err, "OK" if err < 3e-3 else "FAIL"))
    # transient dequant kernel (large-M band feeds cuBLAS)
    for (n, k, bn, bk) in ((4096, 5120, 128, 128), (4096, 5184, 128, 128),
                           (4096, 5120, 64, 64)):
        dev = torch.device("cuda:0")
        x, packed, bscale, y_ref = make_case(n, k, 512, bn, bk, dev)
        wf = ext.qpn8_blk_dequant(packed, bscale, n, k, bn, bk)
        y = torch.nn.functional.linear(x, wf)
        err = (y.float() - y_ref.float()).norm() / y_ref.float().norm()
        ok = err < 3e-3
        allok &= bool(ok)
        print("N=%5d K=%5d M= 512 bn=%3d bk=%3d dequant   rel_err=%.2e  %s"
              % (n, k, bn, bk, err, "OK" if ok else "FAIL"))
    # partial trailing k-block (K % bk != 0, K % 64 == 0)
    allok &= check(4096, 5184, 8, 128, 128, 4, 1)
    # non-128 geometry (bn=64, bk=64)
    allok &= check(4096, 5120, 8, 64, 64, 16, 3)
    print("ALL OK" if allok else "FAILURES PRESENT")
    if allok:
        for m in (1, 8):
            bench(4096, 5120, m, 16, 3)
        bench(5120, 1536, 8, 16, 3)
    sys.exit(0 if allok else 1)
