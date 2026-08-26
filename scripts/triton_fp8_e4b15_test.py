"""Can Triton's pre-Hopper fp8e4b15 stand in for e4m3fn on sm70/sm75?

e4m3fn and e4b15 share the 1-4-3 bit layout and differ only in exponent
bias (7 vs 15), so a bitcast plus a factor 2**8 should reproduce e4m3fn
exactly -- except wherever the two formats disagree on specials.
Checked exhaustively over all 256 byte patterns.
"""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _decode_e4b15(src, dst, N: tl.constexpr):
    i = tl.arange(0, N)
    b = tl.load(src + i)
    v = b.to(tl.float8e4b15, bitcast=True).to(tl.float32)
    tl.store(dst + i, v * 256.0)


@triton.jit
def _encode_e4b15(src, dst, N: tl.constexpr):
    i = tl.arange(0, N)
    v = tl.load(src + i)
    q = (v * (1.0 / 256.0)).to(tl.float8e4b15)
    tl.store(dst + i, q.to(tl.uint8, bitcast=True))


cap = torch.cuda.get_device_capability()
print(f"device={torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}")

# --- decode: all 256 byte patterns ---
codes = torch.arange(256, device="cuda", dtype=torch.uint8)
ref = codes.view(torch.float8_e4m3fn).to(torch.float32)
got = torch.zeros(256, device="cuda", dtype=torch.float32)
try:
    _decode_e4b15[(1,)](codes, got, N=256)
    torch.cuda.synchronize()
except Exception as exc:
    print("decode: COMPILE FAIL --", str(exc).strip().splitlines()[-1][:200])
    raise SystemExit(1)

finite = torch.isfinite(ref)
exact = (got == ref) | (~finite & ~torch.isfinite(got))
bad = (~exact).nonzero().flatten().tolist()
print(f"decode: {int(exact.sum())}/256 byte patterns identical")
if bad:
    print(f"  mismatching codes ({len(bad)}): {bad[:20]}")
    for c in bad[:8]:
        print(f"    0x{c:02X}: e4m3fn={ref[c].item()!r:>12}  e4b15*256={got[c].item()!r}")

# --- encode: round-trip every finite e4m3 value (padded to a power of 2) ---
vals = torch.zeros(256, device="cuda", dtype=torch.float32)
finite_idx = finite.nonzero().flatten()
vals[: finite_idx.numel()] = ref[finite_idx]
out = torch.zeros(256, device="cuda", dtype=torch.uint8)
try:
    _encode_e4b15[(1,)](vals, out, N=256)
    torch.cuda.synchronize()
    back = out.view(torch.float8_e4m3fn).to(torch.float32)
    n = finite_idx.numel()
    ok = int((back[:n] == vals[:n]).sum())
    print(f"encode round-trip: {ok}/{n} finite values exact")
    if ok != n:
        m = (back[:n] != vals[:n]).nonzero().flatten()[:8].tolist()
        for i in m:
            print(f"    in={vals[i].item()!r} -> byte=0x{out[i].item():02X} -> {back[i].item()!r}")
except Exception as exc:
    print("encode: FAIL --", " / ".join(l.strip() for l in str(exc).splitlines() if l.strip())[:300])
