"""Chunked QPN8 across M, including ragged tails.

Production raises the chunked band from M<=16 to M<=96 because chunking beats
the reconstruct path all the way to M~112 (benchmarks/qpn8_m_sweep.py). That
newly exercises RAGGED chunk tails -- M=20 is 8+8+4, M=17 is 8+8+1 -- which
the old M<=16 band only ever hit as a single 9..16 split. This checks every
such M against the fp8->fp16 reference.
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

SHAPES = [(4096, 5120, 16, 2, "gdn in_proj_qkvz"),
          (5120, 1536, 8, 2, "out_proj"),
          (3584, 5120, 16, 2, "attn qkv")]
MS = (8, 9, 12, 16, 17, 20, 33, 40, 64, 96)
SCALE = 0.000972203


def chunked(ext, x, packed, tscale, n, splitk, nacc):
    outs = [ext.gemm_qpn8(x[i:i + 8].contiguous(), packed, tscale, n,
                          splitk, nacc) for i in range(0, x.shape[0], 8)]
    return torch.cat(outs, 0) if len(outs) > 1 else outs[0]


dev = torch.device("cuda:0")
allok = True
for (n, k, splitk, nacc, name) in SHAPES:
    torch.manual_seed(0)
    w8 = torch.randint(0, 256, (n, k), dtype=torch.uint8, device=dev)
    w8[w8 == 0x7F] = 0x7E
    w8[w8 == 0xFF] = 0xFE
    ref_w = (w8.view(torch.float8_e4m3fn).float() * SCALE).half()
    packed = qpn8_prepack(w8)
    tscale = torch.full((n // 32,), SCALE * 256.0, dtype=torch.float32,
                        device=dev)
    print(f"\n=== {name}  N={n} K={k} split={splitk} nacc={nacc} ===")
    for m in MS:
        x = torch.randn(m, k, device=dev, dtype=torch.half) * 0.1
        y_ref = (x.float() @ ref_w.float().t()).half()
        y = chunked(ext, x, packed, tscale, n, splitk, nacc)
        err = (y.float() - y_ref.float()).norm() / y_ref.float().norm()
        ok = err < 3e-3 and y.shape == (m, n)
        allok &= ok
        tail = m % 8
        print(f"  M={m:3d}  chunks={-(-m // 8):2d}"
              f"{'  tail=' + str(tail) if tail else '  (exact)':>12}"
              f"  rel_err={err:.2e}  {'OK' if ok else 'FAIL'}")
print("\nALL OK" if allok else "\nFAILURES PRESENT")
if not allok:
    sys.exit(1)
