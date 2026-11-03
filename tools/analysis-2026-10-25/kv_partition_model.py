"""Rechenmodell für den GLM-5.3-PP5-KV-Pool (aus dem Boot vom 24.10. 21:31 abgeleitet).

Annahmen (jede aus Messzeilen belegt):
- MLA-Schichten i % 4 == 3 (i < 45), alle anderen KDA. Gruppenzahl G = max(cdiv(KDA, MLA) gesamt, je Stufe cdiv(KDA_s, MLA_s))
  (`_pp_balanced_mamba_group_count`).
- Pool-Block je Stufe = MLA_s * 4_792_320 + MLA_s * 304_128 (+ Zusatzslots des Entwurfs auf der letzten Stufe).
- Feste Blöcke je Anfrage = G * (2 + k) + 1 (Indexer-Tail) [+ 5 Fenster-Seiten des Entwurfs]; Kontext = ceil(ctx / 9216) MLA-Blöcke.
- Block-IDs sind global: N = min über Stufen von kv_s / pool_s.
- kv_s = requested - weights - activation - warmup - non_torch - graph.
"""
from itertools import product
from math import ceil

MLA_PAGE = 4_792_320
IDX_PAGE = 304_128
BLOCK_TOKENS = 9_216
LAYERS = 45
MLA = {i for i in range(LAYERS) if i % 4 == 3}
REQ_RTX, REQ_V100 = 49_739_460_772, 33_391_108_424
ACTIVATION = 1_157_711_872
WARMUP = 50_000_000
NON_TORCH = 15_000_000
MOE_LAYER = 4_164_000_000      # Stufe 1-3: 29,15 GB / 7 Schichten
DENSE_LAYER = 190_000_000      # Schichten 0-2 (aus 13-Schicht-Stufe 0)
EMBED = 1_360_000_000          # Einbettung auf Stufe 0
LM_HEAD = 900_000_000          # lm_head auf Stufe 4 (gemessen 46,70 GB bei 11 Schichten)
DRAFTER = 4_060_000_000        # DFlash2-Entwurf + replizierte Einbettung (aus 46,60 GB - 10 Schichten - lm_head)
DRAFT_AUX_LAYERS = 5
DRAFT_WINDOW_PAGES = 5


def layer_weight(i):
    return DENSE_LAYER if i < 3 else MOE_LAYER


def model(partition, k, graph_v100=1_157_711_872, drafter=DRAFTER, activation=ACTIVATION, gmu_extra=0):
    bounds = [0]
    for n in partition:
        bounds.append(bounds[-1] + n)
    assert bounds[-1] == LAYERS
    stages = []
    n_kda_total = LAYERS - len(MLA)
    groups = ceil(n_kda_total / len(MLA))
    for s in range(5):
        layers = range(bounds[s], bounds[s + 1])
        n_mla = sum(i in MLA for i in layers)
        n_kda = len(layers) - n_mla
        if n_kda and not n_mla:
            return None
        if n_kda:
            groups = max(groups, ceil(n_kda / n_mla))
        stages.append((layers, n_mla, n_kda))
    result = []
    for s, (layers, n_mla, n_kda) in enumerate(stages):
        rtx = s in (0, 4)
        weights = sum(layer_weight(i) for i in layers)
        if s == 0:
            weights += EMBED
        if s == 4:
            weights += LM_HEAD + drafter
        requested = (REQ_RTX if rtx else REQ_V100) + gmu_extra * (1 if rtx else 0)
        kv = requested - weights - activation - WARMUP - NON_TORCH - (0 if rtx else graph_v100)
        pool = n_mla * (MLA_PAGE + IDX_PAGE)
        if s == 4:
            pool += max(0, DRAFT_AUX_LAYERS - n_mla) * MLA_PAGE
        result.append((kv, pool, n_mla, n_kda))
    blocks = min(kv // pool for kv, pool, *_ in result if pool)
    fixed = groups * (2 + k) + 1 + DRAFT_WINDOW_PAGES
    ctx_blocks = blocks - fixed
    return groups, blocks, fixed, ctx_blocks * BLOCK_TOKENS, result


def report(partition, k, **kw):
    out = model(partition, k, **kw)
    if out is None:
        return None
    groups, blocks, fixed, ctx, result = out
    kvs = " ".join(f"{kv / 1e9:5.2f}" for kv, *_ in result)
    return f"{partition} k={k} G={groups} N={blocks} fest={fixed} ctx={ctx:>9,d}  kv_GB=[{kvs}]"


if __name__ == "__main__":
    print("Prüfung gegen Messung (14,7,7,7,10, k=7): erwartet G=6, N≈59, kv0≈0,91 GB, kv4≈1,90 GB")
    print(report((14, 7, 7, 7, 10), 7))
    print()
    for k in (7, 5, 4, 3):
        for gr in (1_157_711_872, 65_000_000):
            print(f"-- k={k}, V100-Graph {'1,16 GB (jetzt)' if gr > 1e9 else '0,06 GB (gemessen)'}")
            rows = []
            for a, b, c, d in product(range(8, 17), range(4, 9), range(4, 9), range(4, 9)):
                e = LAYERS - a - b - c - d
                if not 6 <= e <= 13:
                    continue
                out = model((a, b, c, d, e), k, graph_v100=gr)
                if out is None:
                    continue
                groups, blocks, fixed, ctx, result = out
                if min(kv for kv, *_ in result) < 0:
                    continue
                rows.append((ctx, (a, b, c, d, e), groups))
            rows.sort(reverse=True)
            for ctx, part, groups in rows[:4]:
                print("  ", report(part, k, graph_v100=gr))
