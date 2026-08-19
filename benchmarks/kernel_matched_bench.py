"""QPN2/QPN vs Marlin on the shapes and bands we actually dispatch.

Why this exists, and what an earlier version of it got wrong.

The README headlined 5.9x, which divided 565.2 GB/s (five shapes,
kernel_bw_bench.py, 08-11) by 96.0 GB/s (a DIFFERENT shape set, a harness not
in this repo, 08-05). Never measured together. Neither committed harness
measures Marlin at all: kernel_bw_bench.py carries
`fn = None  # marlin measured separately below` with no such measurement, and
skinny_bench.py treats 96 as a constant.

The first attempt to fix that measured `gemm_simt` / `gemm_qpn` / `gemm_wmma`
-- the OLD dispatch. None of those is what serves. From the shipped shim
(fork_patches/marlin.py:209) and a production boot's route census:

    qpn2    owns M 1..8    (including lm_head, N=62080)
    qpn     owns M 9..16
    marlin  owns M >= 17   -- CONCEDED BY DESIGN, not a loss

So a "we lose at M=64" result is meaningless: we do not dispatch there.

This harness measures the bands we actually own, on the shapes we actually
serve (the keys of the shim's own _QPN2_TABLE, which includes the lm_head
projection), and imports the SHIPPED _qpn_prepack and _qpn2_cfg rather than
reimplementing them -- so the thing measured is the thing that runs.

Marlin is reachable on SM70 only through the fork's
VLLM_SM70_QUANT_BACKEND=marlin override ("forces_marlin() => SM70 allowed").
Upstream vLLM declines NVFP4 below compute capability 75 and raises at engine
construction, so this is the fork's fallback, never upstream's.
"""
import os
import sys
import time

import torch

dev = "cuda"
# 825 GB/s is a measured *memcpy* rate: it moves each byte twice (read AND
# write). These GEMMs are read-dominated -- they stream weights in and write a
# tiny MxN result -- so a percentage against this denominator OVERSTATES how
# close to roofline the kernel is, and can exceed 100% (lm_head measures 102%).
# Report the raw GB/s as the primary figure; treat the percentage as a rough
# upper-bound indicator, not a roofline fraction.
COPY_CEIL = float(os.environ.get("COPY_CEILING_GBS", "825"))
READ_CEIL = float(os.environ.get("READ_CEILING_GBS", "0")) or None
GSCALE = 1.0

# The shipped shim: its prepack, its config table, its extension loader.
from vllm.model_executor.kernels.linear.nvfp4 import marlin as shim  # noqa: E402

SHAPES = sorted(shim._QPN2_TABLE.keys())          # (k, n), real serving shapes


def measure_ceilings():
    """Measure BOTH bandwidth conventions, so percentages are unambiguous.

    The 825 GB/s figure this repo has always quoted is a *memcpy* rate, and a
    memcpy touches DRAM twice per element (read + write). These GEMMs stream
    weights IN and write a tiny MxN result OUT, so they are read-dominated.
    Quoting a read-dominated kernel as a percentage of a copy rate is not
    apples-to-apples and can exceed 100% -- lm_head measured 102%, which is a
    denominator artefact, not a physical impossibility.

    read_GBs  : bytes READ / time, from a pure streaming reduction.
    copy_GBs  : bytes READ+WRITTEN / time, from a device-to-device copy.
    """
    n = 512 * 1024 * 1024 // 2                     # 512 MiB of fp16
    a = torch.empty(n, dtype=torch.float16, device=dev).normal_()
    b = torch.empty_like(a)
    nbytes = a.numel() * a.element_size()

    def _t(fn, it=30):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(it):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / it

    t_read = _t(lambda: torch.sum(a))
    t_copy = _t(lambda: b.copy_(a))
    read_gbs = nbytes / t_read / 1e9
    copy_gbs = 2 * nbytes / t_copy / 1e9           # read + write
    del a, b
    torch.cuda.empty_cache()
    return read_gbs, copy_gbs


def bench(fn, it=200, warm=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / it


def packed_bytes(k, n):
    return n * (k // 2) + n * (k // 16)


def marlin_arm():
    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            apply_fp4_marlin_linear, prepare_fp4_layer_for_marlin,
            is_fp4_marlin_supported)
    except Exception as exc:
        print(f"# marlin unavailable: {exc}", file=sys.stderr)
        return None
    print(f"# is_fp4_marlin_supported() -> {is_fp4_marlin_supported()}"
          f"  (VLLM_SM70_QUANT_BACKEND={os.environ.get('VLLM_SM70_QUANT_BACKEND')})")
    if not is_fp4_marlin_supported():
        return None

    class _L(torch.nn.Module):
        pass

    def build(k, n, codes, sbytes):
        lay = _L()
        lay.params_dtype = torch.float16
        lay.weight = torch.nn.Parameter(codes.clone(), requires_grad=False)
        lay.weight_scale = torch.nn.Parameter(
            sbytes.clone().view(torch.float8_e4m3fn), requires_grad=False)
        lay.weight_global_scale = torch.nn.Parameter(
            torch.tensor([GSCALE], device=dev), requires_grad=False)
        lay.input_size_per_partition = k
        lay.output_size_per_partition = n
        prepare_fp4_layer_for_marlin(lay)
        return lay

    def call(lay, x, k, n):
        return apply_fp4_marlin_linear(
            input=x, weight=lay.weight, weight_scale=lay.weight_scale,
            weight_global_scale=lay.weight_global_scale,
            workspace=lay.workspace, size_n=n, size_k=k, bias=None)

    return build, call


def qpn8_arm():
    """FP8 W8A16 arm. Serves M 1..96 on the mixed checkpoint's 128 protected
    layers -- a wider band than QPN2's 1..8, and absent from every kernel
    table we publish. Uses the SHIPPED prepack and config table."""
    try:
        from vllm.model_executor.layers.quantization import modelopt as mo
    except Exception as exc:
        print(f"# qpn8 unavailable: {exc}", file=sys.stderr)
        return None
    if not hasattr(mo, "_sm70_qpn8_prepack"):
        print("# qpn8 prepack not present in this build", file=sys.stderr)
        return None
    return mo


# The real per-rank shapes, split by role. lm_head is EXCLUDED from the trunk
# aggregate: at 5120x62080 it is ~160 MB packed against a few MB per trunk
# projection, so a byte-weighted aggregate that includes it is essentially the
# lm_head number alone. It is reported separately instead.
LM_HEAD = (5120, 62080)
FP8_SHAPES = [(1536, 5120), (5120, 3584), (5120, 4096)]   # (k, n), QPN8


def main():
    ext = shim._get_skinny_ext()
    if ext is None:
        sys.exit("skinny extension failed to build")
    read_gbs, copy_gbs = measure_ceilings()
    print(f"# measured ceilings: read-only {read_gbs:.0f} GB/s, "
          f"copy(read+write) {copy_gbs:.0f} GB/s")
    print(f"# percentages below use the READ ceiling ({read_gbs:.0f} GB/s), "
          f"which is the right denominator for a read-dominated GEMM.")
    global COPY_CEIL
    COPY_CEIL = read_gbs
    marlin = marlin_arm()
    mo = qpn8_arm()
    g = torch.Generator(device="cpu").manual_seed(0)

    def build_nvfp4(k, n):
        codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8,
                              generator=g).to(dev)
        sbytes = torch.randint(0x30, 0x50, (n, k // 16), dtype=torch.uint8,
                               generator=g).to(dev)
        qc, qs = shim._qpn_prepack(codes, sbytes)
        lay = marlin[0](k, n, codes, sbytes) if (marlin and qc is not None) else None
        return qc, qs, lay, packed_bytes(k, n), shim._qpn2_cfg(k, n)

    trunk = [(k, n) + build_nvfp4(k, n) for k, n in SHAPES if (k, n) != LM_HEAD]
    trunk = [t for t in trunk if t[2] is not None]
    head = [(k, n) + build_nvfp4(k, n) for k, n in SHAPES if (k, n) == LM_HEAD]
    head = [t for t in head if t[2] is not None]

    def run(group, M):
        route = "qpn2" if M <= 8 else "qpn"
        to = tm = tb = 0.0
        for k, n, qc, qs, lay, gb, cfg in group:
            x = torch.randn(M, k, dtype=torch.float16, device=dev) * 0.5
            if route == "qpn2":
                if cfg is None:
                    return None
                f = lambda: ext.gemm_qpn2(x, qc, qs, GSCALE, n, cfg[0], cfg[1])
            else:
                f = lambda: ext.gemm_qpn(x, qc, qs, GSCALE, n)
            to += bench(f)
            if marlin and lay is not None:
                tm += bench(lambda: marlin[1](lay, x, k, n))
            tb += gb
        o = tb / (to * 1e-3) / 1e9
        m = (tb / (tm * 1e-3) / 1e9) if (marlin and tm > 0) else None
        return route, o, m

    print("# NVFP4 arm: QPN2 (M 1-8) / QPN (M 9-16) vs the fork's Marlin "
          "fallback. M>=17 is Marlin by design.")
    print("# TRUNK shapes only -- lm_head excluded and reported separately.")
    print(f"# trunk = {[(k, n) for k, n, *_ in trunk]}")
    print("group,M,route,ours_GBs,ours_pct,marlin_GBs,marlin_pct,ratio")
    for label, group in (("trunk", trunk), ("lm_head", head)):
        if not group:
            continue
        for M in (1, 2, 4, 8, 16):
            r = run(group, M)
            if r is None:
                continue
            route, o, m = r
            if m:
                print(f"{label},{M},{route},{o:.1f},{o / COPY_CEIL * 100:.0f}%,"
                      f"{m:.1f},{m / COPY_CEIL * 100:.0f}%,{o / m:.2f}", flush=True)
            else:
                print(f"{label},{M},{route},{o:.1f},{o / COPY_CEIL * 100:.0f}%,"
                      f"NA,NA,NA", flush=True)

    # ---- QPN8, the FP8 half nothing has ever measured ---------------------
    if mo is None:
        print("# QPN8 arm skipped")
    else:
        print("\n# QPN8 (FP8 W8A16) on the 128 protected layers. Serves M 1-96.")
        print(f"# shapes = {FP8_SHAPES}")
        print("group,M,route,ours_GBs,ours_pct")
        f8 = []
        for k, n in FP8_SHAPES:
            w8 = torch.randint(0x30, 0x50, (n, k), dtype=torch.uint8,
                               generator=g).to(dev)
            packed = mo._sm70_qpn8_prepack(w8)
            # one scale per N=32 tile, float32, contiguous -- the kernel
            # asserts all three.
            ts = torch.ones(n // 32, dtype=torch.float32,
                            device=dev).contiguous()
            cfg = mo._SM70_QPN8_TABLE.get((n, k), (16, 2))
            mt2 = mo._SM70_QPN8_MT2_TABLE.get((n, k), cfg)
            f8.append((k, n, packed, ts, cfg, mt2, n * k))
        for M in (1, 2, 4, 8, 16, 32, 64):
            to = tb = 0.0
            route = "qpn8" if M <= 8 else ("qpn8-mt2" if M <= 16 else "qpn8-chunked")
            for k, n, packed, ts, cfg, mt2, gb in f8:
                x = torch.randn(M, k, dtype=torch.float16, device=dev) * 0.5
                if M <= 8:
                    f = lambda: ext.gemm_qpn8(x, packed, ts, n, cfg[0], cfg[1])
                elif M <= 16 and hasattr(ext, "gemm_qpn8_mt2"):
                    f = lambda: ext.gemm_qpn8_mt2(x, packed, ts, n, mt2[0], mt2[1])
                else:
                    def f(x=x, k=k, n=n, packed=packed, ts=ts, cfg=cfg):
                        return torch.cat([ext.gemm_qpn8(x[i:i + 8].contiguous(),
                                                        packed, ts, n, cfg[0], cfg[1])
                                          for i in range(0, x.shape[0], 8)])
                to += bench(f, it=100)
                tb += gb
            o = tb / (to * 1e-3) / 1e9
            print(f"fp8,{M},{route},{o:.1f},{o / COPY_CEIL * 100:.0f}%", flush=True)
    print("MATCHED_BENCH_DONE")


if __name__ == "__main__":
    main()
