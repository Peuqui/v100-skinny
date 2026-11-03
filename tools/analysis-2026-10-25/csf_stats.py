"""Decode csf_32x2_v1 planes of some GLM expert matrices on the CPU (format as csf32_expand_sm70.cu reads it) and measure
how many scales a finer base granularity would still leave as exceptions."""
import json, sys, torch
from safetensors import safe_open
root = "/home/mp/models/GLM-5.3-Flash-NVFP4-MTP"
idx = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]

def get(base, part):
    with safe_open(f"{root}/{idx[base + part]}", framework="pt") as f:
        return f.get_tensor(base + part)

def decode(base, rows, groups):
    bases = get(base, "csf32_bases").view(torch.uint8).reshape(rows)
    codes = get(base, "csf32_codes").view(torch.uint8).reshape(-1)
    page_bases = get(base, "csf32_page_bases").view(torch.int32).reshape(-1)
    offsets = (get(base, "csf32_offsets").view(torch.int16).to(torch.int32) & 0xFFFF).reshape(-1)
    exc = (get(base, "csf32_exceptions").view(torch.int16).to(torch.int32) & 0xFFFF).reshape(-1)
    tile_rows, tile_pairs = rows // 128, groups // 4
    # codes index: ((tile_row*tile_pairs + tile_pair)*8 + at)*32 + row_in_block, at = half*4 + bir
    c = codes.view(tile_rows, tile_pairs, 2, 4, 32)           # [tr, tp, half, bir, r32]
    c = c.permute(0, 3, 4, 1, 2).reshape(rows, tile_pairs * 2)  # [row = tr*128 + bir*32 + r32, group pair gp = tp*2 + half]
    scales = torch.empty(rows, groups, dtype=torch.int32)
    base_i = bases.to(torch.int32)[:, None]
    scales[:, 0::2] = (base_i + (c & 0xF).to(torch.int32)) & 0xFF
    scales[:, 1::2] = (base_i + (c >> 4).to(torch.int32)) & 0xFF
    exc_mask = torch.zeros(rows, groups, dtype=torch.bool)
    n_pages = page_bases.numel()
    for page in range(n_pages):
        seg = offsets[page * 257:(page + 1) * 257]
        origin = int(page_bases[page]) >> 1
        for b in range(seg.numel() - 1):
            first, last = int(seg[b]), int(seg[b + 1])
            if first == last: continue
            block = page * 256 + b
            tr_tp, at = divmod(block, 8)
            tr, tp = divmod(tr_tp, tile_pairs)
            half, bir = divmod(at, 4)
            for j in range(first, last):
                rec = int(exc[origin + j]); row_in = (rec & 0xFF) >> 1; col = rec & 1; val = rec >> 8
                r = tr * 128 + bir * 32 + row_in
                g = (tp * 2 + half) * 2 + col
                scales[r, g] = val; exc_mask[r, g] = True
    return scales, exc_mask, bases

layer, e = int(sys.argv[1]), int(sys.argv[2])
for proj, rows, groups in (("gate_proj", 2048, 256), ("up_proj", 2048, 256), ("down_proj", 4096, 128)):
    base = f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}.weight_scale."
    s, m, b = decode(base, rows, groups)
    delta = (s - b.to(torch.int32)[:, None]) & 0xFF
    non_exc = ~m
    ok = bool((delta[non_exc] <= 15).all())
    print(f"{proj}: Skalen {s.min().item()}..{s.max().item()}, Ausnahmen {int(m.sum())} ({100*m.float().mean():.3f} %), "
          f"alle übrigen im Fenster 0..15: {ok}")
    for seg in (256, 64, 32, 16, 8):
        if seg > groups: continue
        view = s.view(rows, groups // seg, seg)
        lo = view.min(dim=2).values
        spread = (view - lo[..., None]).max()
        outside = ((view - lo[..., None]) > 15).sum().item()
        # window start free (best base): count scales outside [lo, lo+15]
        print(f"   Segment {seg:3d} Gruppen: Spannweite max {int(spread)}, außerhalb Fenster ab Minimum: {outside} ({100*outside/s.numel():.4f} %)")
