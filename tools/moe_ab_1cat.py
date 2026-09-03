#!/usr/bin/env python3
"""A/B: 1Cat-vLLM 1.5.0 compact-grouped NVFP4-MoE (TurboMind-Ops) gegen
v100-skinny moe_qpn, auf echten Qwen3.6-35B-A3B-NVFP4-Expert-Bytes (Layer 1,
E=256, topk=8, w13 1024x2048, w2 2048x512 — 1Cats kleinster Contract).

Laeuft in .venv-sm70-150 (1.5.0-Wheel). Nur V100: 1Cats apply() ist
exakt-sm70-gegated; die Ops selbst werden hier direkt gerufen. Gemessen
werden die reinen Gewichts-GEMM-Stufen (w13 + w2) beider Routen bei
IDENTISCHEM, vorab praeparierten Routing (perm/offsets ausserhalb der
Messschleife — beide Seiten gleich behandelt). Korrektheit: beide Routen
gegen eine unabhaengige fp16-Dequant-Referenz (torch-Einsum).

Aufruf: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  .venv-sm70-150/bin/python tools/moe_ab_1cat.py
"""
import sys
import time

import torch

MODELS = {
    "qwen36": dict(
        shard="/home/mp/models/Qwen3.6-35B-A3B-NVFP4/model-00001-of-00003.safetensors",
        prefix="model.language_model.layers.1.mlp.experts.{e}",
        names=("gate_proj", "up_proj", "down_proj"), E=256, topk=8),
    "dsv4": dict(
        shard="/home/mp/models/DeepSeek-V4-Flash-nvfp4-DSpark/model-00007-of-00046.safetensors",
        prefix="layers.5.ffn.experts.{e}",
        names=("w1", "w3", "w2"), E=64, topk=6),
}
CFG = MODELS[sys.argv[1] if len(sys.argv) > 1 else "qwen36"]
LAYERTAG = CFG["prefix"]
E, TOPK = CFG["E"], CFG["topk"]
SKINNY_SRC = "/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu"

# ---------------------------------------------------------------- Laden
from safetensors import safe_open

g13g, g13u, g2g = [], [], []
w13_c, w13_s, w2_c, w2_s = [], [], [], []
GATE, UP, DOWN = CFG["names"]
with safe_open(CFG["shard"], framework="pt", device="cuda") as f:
    for e in range(E):
        b = LAYERTAG.format(e=e)
        w13_c.append(torch.cat([f.get_tensor(f"{b}.{GATE}.weight"),
                                f.get_tensor(f"{b}.{UP}.weight")], 0).contiguous())
        w13_s.append(torch.cat([f.get_tensor(f"{b}.{GATE}.weight_scale"),
                                f.get_tensor(f"{b}.{UP}.weight_scale")], 0).contiguous())
        g13g.append(float(f.get_tensor(f"{b}.{GATE}.weight_scale_2").item()))
        g13u.append(float(f.get_tensor(f"{b}.{UP}.weight_scale_2").item()))
        w2_c.append(f.get_tensor(f"{b}.{DOWN}.weight").contiguous())
        w2_s.append(f.get_tensor(f"{b}.{DOWN}.weight_scale").contiguous())
        g2g.append(float(f.get_tensor(f"{b}.{DOWN}.weight_scale_2").item()))
w13_c, w13_s, w2_c, w2_s = map(torch.stack, (w13_c, w13_s, w2_c, w2_s))
w13_su, w2_su = w13_s.view(torch.uint8), w2_s.view(torch.uint8)
N13, K13 = w13_c.size(1), w13_c.size(2) * 2
N2, K2 = w2_c.size(1), w2_c.size(2) * 2
INTER = N13 // 2
same_g = all(abs(a - b) < 1e-12 for a, b in zip(g13g, g13u))
print(f"device: {torch.cuda.get_device_name()}")
print(f"w13 {N13}x{K13}, w2 {N2}x{K2}, E={E}, topk={TOPK}, "
      f"gate/up-globals identisch: {same_g}")

# ------------------------------------------- fp16-Dequant-Referenz (torch)
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                     dtype=torch.float32, device="cuda")


def dequant(codes_u8, scales_f8, gvec):
    """[N, K/2] codes + [N, K/16] fp8-Scales + Global je ZEILE -> fp32 [N, K]."""
    n, k2 = codes_u8.shape
    nib = torch.stack([codes_u8 & 0xF, codes_u8 >> 4], -1).view(n, k2 * 2).long()
    vals = _E2M1[nib]
    sc = scales_f8.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    return vals * sc * gvec.view(-1, 1)


def ref_moe(hs, ids, w13_dq, w2_dq):
    """Referenz: fp32-Einsums auf dequantisierten Gewichten, silu*up."""
    T = hs.size(0)
    out13 = torch.empty(T * TOPK, N13, dtype=torch.float32, device="cuda")
    out = torch.empty(T * TOPK, N2, dtype=torch.float32, device="cuda")
    flat = ids.reshape(-1)
    for s in range(T * TOPK):
        e = int(flat[s])
        x = hs[s // TOPK].float()
        y13 = x @ w13_dq[e].t()
        out13[s] = y13
        mid = torch.nn.functional.silu(y13[:INTER]) * y13[INTER:]
        out[s] = mid.half().float() @ w2_dq[e].t()
    return out13, out


# ---------------------------------------------------- unsere Seite (skinny)
import os

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
from torch.utils.cpp_extension import load

ext = load(name="skinny_nvfp4_ab", sources=[SKINNY_SRC],
           extra_cuda_cflags=["-O3", "--use_fast_math",
                              "-gencode=arch=compute_70,code=sm_70"])


def qpn_prepack(codes, scales):
    """Fragment-Order-Prepack (Kopie aus fork_patches/marlin.py)."""
    n, k2 = codes.shape
    k = k2 * 2
    dev = codes.device
    tiles, groups = n // 32, k // 16
    lane = torch.arange(32, device=dev)
    col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) > 0).long() * 4
    korder = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7,
                           8, 10, 12, 14, 9, 11, 13, 15], device=dev)
    nib = torch.stack([codes & 0xF, codes >> 4], dim=-1).view(n, k)
    g = torch.arange(groups, device=dev)
    kidx = g.view(groups, 1) * 16 + korder.view(1, 16)
    qc = torch.empty(tiles, groups, 32, 8, dtype=torch.uint8, device=dev)
    qs = torch.empty(tiles, groups, 32, dtype=torch.uint8, device=dev)
    ncol = (torch.arange(tiles, device=dev).view(tiles, 1) * 32 + col.view(1, 32))
    nb = nib[ncol.view(tiles, 1, 32, 1).expand(tiles, groups, 32, 16),
             kidx.view(1, groups, 1, 16).expand(tiles, groups, 32, 16)]
    qc[:] = nb[..., 0::2] | (nb[..., 1::2] << 4)
    qs[:] = scales[ncol.view(tiles, 1, 32).expand(tiles, groups, 32),
                   g.view(1, groups, 1).expand(tiles, groups, 32)]
    return qc.view(-1).contiguous(), qs.view(-1).contiguous()


p13 = [qpn_prepack(w13_c[e], w13_su[e]) for e in range(E)]
p2 = [qpn_prepack(w2_c[e], w2_su[e]) for e in range(E)]
qc13 = torch.stack([p[0] for p in p13]).contiguous()
qs13 = torch.stack([p[1] for p in p13]).contiguous()
qc2 = torch.stack([p[0] for p in p2]).contiguous()
qs2 = torch.stack([p[1] for p in p2]).contiguous()
del p13, p2
g13_t = torch.tensor(g13g, dtype=torch.float32, device="cuda")
g13u_t = torch.tensor(g13u, dtype=torch.float32, device="cuda")
g2_t = torch.tensor(g2g, dtype=torch.float32, device="cuda")
# gate/up-globals ungleich -> zwei Aufrufe auf N-Haelften (eigene Packs)
if not same_g:
    half = N13 // 2
    pg = [qpn_prepack(w13_c[e, :half], w13_su[e, :half]) for e in range(E)]
    pu = [qpn_prepack(w13_c[e, half:], w13_su[e, half:]) for e in range(E)]
    qc13g = torch.stack([p[0] for p in pg]).contiguous()
    qs13g = torch.stack([p[1] for p in pg]).contiguous()
    qc13u = torch.stack([p[0] for p in pu]).contiguous()
    qs13u = torch.stack([p[1] for p in pu]).contiguous()
    del pg, pu

# ------------------------------------------------- 1Cat-Seite (TurboMind)
from vllm import _sm70_ops as sm70_ops
from vllm.model_executor.layers.quantization.sm70_turbomind import (
    NVFP4_GROUP_SIZE,
    unpack_mxfp4_weight,
)

tm13_w, tm13_s, tm13_m, tm2_w, tm2_s, tm2_m = [], [], [], [], [], []
for e in range(E):
    packed = unpack_mxfp4_weight(w13_c[e])
    sc = w13_s[e].float().clone()
    sc[:INTER].mul_(g13g[e])
    sc[INTER:].mul_(g13u[e])
    pw, ps, pm = sm70_ops.nvfp4_sm70_prepare(
        packed, sc.half().t().contiguous(), NVFP4_GROUP_SIZE,
        interleave_gated_silu=False)
    tm13_w.append(pw)
    tm13_s.append(ps)
    tm13_m.append(pm)
    packed = unpack_mxfp4_weight(w2_c[e])
    sc = w2_s[e].float() * g2g[e]
    pw, ps, pm = sm70_ops.nvfp4_sm70_prepare(
        packed, sc.half().t().contiguous(), NVFP4_GROUP_SIZE)
    tm2_w.append(pw)
    tm2_s.append(ps)
    tm2_m.append(pm)
tm13_w = torch.stack(tm13_w)
tm13_s = torch.stack(tm13_s)
tm2_w = torch.stack(tm2_w)
tm2_s = torch.stack(tm2_s)
w13_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
    tm13_w, tm13_s, int(tm13_m[0][0].item()), int(tm13_m[0][1].item()), E)
w2_ptrs = sm70_ops.awq_moe_build_strided_ptrs(
    tm2_w, tm2_s, int(tm2_m[0][0].item()), int(tm2_m[0][1].item()), E)
print("1Cat TurboMind prepare ok")


def t(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


print(f"\n{'T':>2} {'slots':>5} {'exp':>4} | {'1cat w13+w2':>12} | "
      f"{'moe_qpn':>8} | ratio | diff 1cat / qpn (vs fp32-Referenz)")
for T in (1, 2, 4, 6, 8):
    torch.manual_seed(T)
    hs = (torch.randn(T, K13, device="cuda", dtype=torch.float16) / 8).contiguous()
    _, ids = torch.topk(torch.randn(T, E, device="cuda"), TOPK, dim=-1)
    ids = ids.to(torch.int32)
    S = T * TOPK
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True).to(torch.int32)
    # --- gemeinsame Routing-Praeparation (ausserhalb der Messung) ---
    # 1Cat compact grouped: 1 Zeile je Gruppe, Slots expert-sortiert
    perm_x = hs[(order.long() // TOPK)].contiguous()          # [S, K13]
    active_ids = flat[order.long()].to(torch.int32).contiguous()
    offsets_1cat = torch.arange(S + 1, dtype=torch.int32, device="cuda")
    # unsere Route: compact-Gruppen (Experten dedupliziert, Padding leer)
    sorted_e = flat[order.long()].to(torch.int64)
    newg = torch.ones(S, dtype=torch.bool, device="cuda")
    newg[1:] = sorted_e[1:] != sorted_e[:-1]
    gidx = torch.cumsum(newg, 0) - 1
    goff = torch.full((S + 1,), S, dtype=torch.int64, device="cuda")
    goff.scatter_reduce_(0, gidx, torch.arange(S, dtype=torch.int64, device="cuda"),
                         reduce="amin")
    gq = torch.zeros(S, dtype=torch.int64, device="cuda")
    gq.scatter_(0, gidx, sorted_e)
    gids_qpn, goff_qpn = gq.to(torch.int32), goff.to(torch.int32)

    gate_up = torch.empty(S, N13, dtype=torch.float16, device="cuda")
    mid = (torch.randn(S, INTER, device="cuda", dtype=torch.float16) / 8).contiguous()
    y2_1cat = torch.empty(S, N2, dtype=torch.float16, device="cuda")

    def run_1cat_w13():
        sm70_ops.nvfp4_moe_dense_stage_sm70_out(
            gate_up, perm_x, offsets_1cat, active_ids,
            w13_ptrs[0], w13_ptrs[1], S, K13, N13, NVFP4_GROUP_SIZE)

    def run_1cat_w2():
        sm70_ops.nvfp4_moe_dense_stage_sm70_out(
            y2_1cat, mid, offsets_1cat, active_ids,
            w2_ptrs[0], w2_ptrs[1], S, K2, N2, NVFP4_GROUP_SIZE)

    y13_qpn = torch.empty(S, N13, dtype=torch.float16, device="cuda")
    y2_qpn = torch.empty(S, N2, dtype=torch.float16, device="cuda")
    # moe_qpn adressiert slot-major ueber die ORIGINAL-Slot-Nummer; mid liegt
    # in 1Cat-Sortierreihenfolge -> zurueckstreuen (gleiche Bytes, fair).
    mid_slotmajor = torch.empty_like(mid)
    mid_slotmajor[order.long()] = mid

    if same_g:
        def run_qpn_w13():
            ext.moe_qpn(hs, qc13, qs13, g13_t, order, gids_qpn, goff_qpn, TOPK,
                        y13_qpn, False, T, 16, 1)
    else:
        half = N13 // 2
        y13g = y13_qpn[:, :half]
        y13u = y13_qpn[:, half:]
        yg = torch.empty(S, half, dtype=torch.float16, device="cuda")
        yu = torch.empty(S, half, dtype=torch.float16, device="cuda")

        def run_qpn_w13():
            ext.moe_qpn(hs, qc13g, qs13g, g13_t, order, gids_qpn, goff_qpn, TOPK,
                        yg, False, T, 16, 1)
            ext.moe_qpn(hs, qc13u, qs13u, g13u_t, order, gids_qpn, goff_qpn, TOPK,
                        yu, False, T, 16, 1)
            y13g.copy_(yg)
            y13u.copy_(yu)

    def run_qpn_w2():
        ext.moe_qpn(mid_slotmajor, qc2, qs2, g2_t, order, gids_qpn, goff_qpn, TOPK,
                    y2_qpn, True, T, 8, 1)

    # --- Korrektheit: beide gegen fp32-Referenz (nur w13-Stufe + w2-Stufe),
    # Referenz nur fuer die tatsaechlich aktiven Experten (Speicher) ---
    act = sorted(set(flat.tolist()))
    dq13 = {e: dequant(w13_c[e], w13_su[e],
                       torch.cat([torch.full((INTER,), g13g[e], device="cuda"),
                                  torch.full((INTER,), g13u[e], device="cuda")]))
            for e in act}
    dq2 = {e: dequant(w2_c[e], w2_su[e],
                      torch.full((N2,), g2g[e], device="cuda")) for e in act}
    run_1cat_w13(); run_1cat_w2()
    run_qpn_w13(); run_qpn_w2()
    e1 = eq = e1b = eqb = 0.0
    for s in range(S):
        e = int(active_ids[s])
        x = perm_x[s].float()
        ref13 = x @ dq13[e].t()
        e1 = max(e1, (gate_up[s].float() - ref13).abs().max().item())
        # qpn: Slot s (sortiert) entspricht y13_qpn[order[s]]
        eq = max(eq, (y13_qpn[order[s].long()].float() - ref13).abs().max().item())
        ref2 = mid[s].float() @ dq2[e].t()
        e1b = max(e1b, (y2_1cat[s].float() - ref2).abs().max().item())
        eqb = max(eqb, (y2_qpn[order[s].long()].float() - ref2).abs().max().item())
    t1 = t(run_1cat_w13) + t(run_1cat_w2)
    tq = t(run_qpn_w13) + t(run_qpn_w2)
    print(f"{T:>2} {S:>5} {len(act):>4} | {t1:>9.3f} ms | {tq:>6.3f} ms | "
          f"{t1/tq:>5.2f}x | w13 {e1:.1e}/{eq:.1e}  w2 {e1b:.1e}/{eqb:.1e}")
