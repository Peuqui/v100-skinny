"""nsys-Auswertung des Decodes von DeepSeek-V4-Flash 284B (PP5, DSpark k = 5, Baum 1Cat-vLLM-v15).

Erweitert analyse_nsys_stufen.py (GLM, ../29-glm-profile-v14k4-2026-11-23/): gleiche Passerkennung und
Rundenlogik, aber mit Rollenklassifikation je Kern (Zustandsautomat je Durchlauf) und Bytes je Rolle aus
den safetensors-Headern. Nur lesen (sqlite im Modus ro), keine GPU.

Aufruf: analyse_nsys_dsv4.py <capture.sqlite> [--json AUS.json]

Zuordnung nsys-deviceId -> Stufe: deviceId 0..4 ist die Stufe (CUDA_VISIBLE_DEVICES=0,1,4,3,2 + PCI_BUS_ID im
Produktionseintrag; die Tabelle TARGET_INFO_GPU nennt dagegen die PCI-Reihenfolge und wird nicht benutzt).
Stufen 0 und 4 sind Quadro RTX 8000 (Turing-Kerne), 1 bis 3 Tesla V100 (Volta-Kerne).
"""
import argparse
import collections
import json
import re
import sqlite3
import statistics as st

PEAK = {0: 672, 1: 900, 2: 900, 3: 900, 4: 672}  # GB/s nominal; praktische Decke 85 %
# Treiberwert memoryBandwidth aus TARGET_INFO_GPU beider Captures (und der GLM-Captures): RTX 624,1 GB/s (Speichertakt unter Last
# 6,5 statt 7,0 GHz), V100 898,0 GB/s. Zusatzspalte, die Hauptzahlen bleiben gegen die nominelle Spitze (vergleichbar mit GLM).
PEAK_DRIVER = {0: 624.1, 1: 898.0, 2: 898.0, 3: 898.0, 4: 624.1}
NAME = {0: "S0 RTX", 1: "S1 V100", 2: "S2 V100", 3: "S3 V100", 4: "S4 RTX"}
TARGET_LAYERS = {0: 10, 1: 8, 2: 8, 3: 8, 4: 9}  # VLLM_PP_LAYER_PARTITION=10,8,8,8,9
FIRST_LAYER = {0: 0, 1: 10, 2: 18, 3: 26, 4: 34}
DRAFT_LAYERS = {4: 3}  # mtp.0..2 auf der letzten Stufe (config: 43 Zielschichten + dspark Entwurf)
# compress_ratios aus config.json: Schicht 0,1: 0; ab 2 gerade = 4 (CSA), ungerade = 128 (HCA); Entwurfsschichten 0
def layer_ratio(global_layer: int) -> int:
    if global_layer >= 43:
        return 0
    if global_layer < 2:
        return 0
    return 4 if global_layer % 2 == 0 else 128

# Bytes je Aufruf (dezimal) aus den Headern (config + safetensors). Gewichte + e8m0-Skalen.
MB = 1e6
EXPERT_UP = 2 * (4194304 + 262144)      # w1 + w3: I8 [2048,2048] + e8m0 [2048,128]
EXPERT_DOWN = 4194304 + 262144          # w2: I8 [4096,1024] + e8m0 [4096,64]
BYTES = {
    "wq_a+wkv": 4194304 + 2097152 + 256 + 128,
    "wq_b": 33554432 + 2048,
    "wo_b": 33554432 + 2048,
    "wo_a_fp8": 33554432 + 2048,        # V100: TurboMind FP8
    "wo_a_fp16": 8192 * 4096 * 2,       # Turing: beim Laden nach FP16 entpackt (fp8.py dequantized_bmm)
    "shared_gu": 2 * (8388608 + 512),
    "shared_dn": 8388608 + 512,
    "router_gate": 256 * 4096 * 2,      # BF16
    "lm_head": 129280 * 4096 * 2,       # BF16
    "main_proj": 50331648 + 3072,
    "indexer_wq_b": 8192 * 1024 + 512,  # Indexer wq_b FP8 [8192,1024] (nur bei Kontext > Top-k, ratio-4-Schichten)
    "markov": 129280 * 256 * 2,         # eine der beiden Matrizen je Schritt (Annahme)
}
SIDE_PRE = re.compile(r"^sm70_glm_mhc_pre_norm_kernel")


def load(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    names = dict(db.execute("select id,value from StringIds"))
    t0 = db.execute("select min(start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    K = {}
    for d in range(5):
        K[d] = [{"s": s, "e": e, "n": names[n], "g": (gx, gy, gz), "b": bx} for s, e, n, gx, gy, gz, bx in db.execute(
            "select start,end,shortName,gridX,gridY,gridZ,blockX from CUPTI_ACTIVITY_KIND_KERNEL "
            "where deviceId=? order by start", (d,))]
    return K, t0


def is_nccl(k):
    return k["n"].startswith("nccl")


def is_moe(k):
    return k["n"].startswith("skinny")


def segments(rows, gap_ms=0.25):
    rows = [k for k in rows if not is_nccl(k)]
    out, cur = [], [rows[0]]
    for r in rows[1:]:
        if (r["s"] - cur[-1]["e"]) / 1e6 > gap_ms:
            out.append(cur)
            cur = [r]
        else:
            cur.append(r)
    out.append(cur)
    return out


def decode_passes(rows):
    """Decode-Durchlaeufe = Segmente mit >= 16 MoE-Kernen auf Gitter y = 36 (6 Zeilen mal Top-6);
    Prefill-Stuecke haben y = 156/256, die kurzen Vorbereitungsdurchlaeufe keinen MoE-Kern."""
    out = []
    for s in segments(rows):
        m = [k for k in s if is_moe(k)]
        if len(m) >= 16 and any(k["g"][1] == 36 for k in m):
            out.append(s)
    return out


def dur(p):
    return (p[-1]["e"] - p[0]["s"]) / 1e6


def busy(p):
    t = last = 0
    for k in p:
        t += max(0, k["e"] - max(k["s"], last))
        last = max(last, k["e"])
    return t / 1e6


def stats(x):
    return f"Mittel {st.mean(x):.2f} min {min(x):.2f} max {max(x):.2f} sd {st.pstdev(x):.2f}"


# ---------------------------------------------------------------- Rollenklassifikation
def classify(d, p):
    """Liefert je Kern (rolle, schicht_global oder None, region). Zustandsautomat:
    Schicht = Zaehlung der mHC-pre_norm-Kerne (ungerade = Attention-Seite, gerade = FFN-Seite);
    Regionen: prep (vor Schicht 0), layer, head1 (ab Kopf 1 bis zur ersten Entwurfsschicht), draft (mtp-Schichten),
    head2, markov."""
    n_target = TARGET_LAYERS[d]
    out = []
    n_pre = moe_count = heads = 0
    region = "prep"
    for k in p:
        n, g, b = k["n"], k["g"], k["b"]
        if SIDE_PRE.match(n):
            n_pre += 1
            if n_pre == 1:
                region = "layer"
            elif d == 4 and n_pre == 2 * n_target + 1:
                region = "draft"
        layer_local = (n_pre - 1) // 2 if n_pre else None
        side = ("attn" if n_pre % 2 == 1 else "ffn") if n_pre else None
        role = None
        if n == "Kernel2" and g[1] == 1010:
            heads += 1
            role = "lm_head"
            region = "head1" if heads == 1 else "head2"
        elif n.startswith("gemv2T") or (n == "reduce_kernel" and g == (1, 1, 1)):
            role, region = "markov_head", "markov"
        elif is_moe(k):
            role = "moe_up" if moe_count % 2 == 0 else "moe_down"
            moe_count += 1
        elif n.startswith(("mhc", "sm70_glm_mhc", "_sm70_mhc", "hc_head")):
            role = "mhc"
        elif n in ("_rejection_kernel", "_resample_kernel", "_compute_block_stats_kernel"):
            role = "sampler"
        elif n.startswith(("_prefill_index_scores", "persistent_topk", "_gather_paged_index_k")):
            role = "indexer"
        elif n.startswith("fp8_qpn8"):
            if region in ("layer", "draft") and side == "attn":
                role = {48: "wq_a+wkv", 1024: "wq_b", 128: "wo_b", 256: "indexer_wq_b"}.get(g[0], "fp8_other")
            elif region in ("layer", "draft") and side == "ffn":
                role = "shared_gu" if b == 512 else "shared_dn"
            elif g == (128, 1, 1) and b == 512:
                role = "main_proj"
            elif g[0] == 48:
                role = "draft_ctx_proj"
            else:
                role = "fp8_other"
        elif (n.startswith(("turing_fp16_s1688gemm", "volta_fp16_s884gemm")) and g[2] == 8) or \
                (n == "gemm_kernel" and g[2] == 2):
            role = "wo_a"
        elif n.startswith(("turing", "volta")):
            role = "attn_qk_pv"
        elif n in ("Kernel2", "splitKreduce_kernel", "gemmSN_TN_kernel"):
            role = "compress_gemm" if side == "attn" else "router_gate"
        elif n in ("topkGatingSoftplusSqrt", "_group_routing_kernel"):
            role = "router_topk"
        elif n == "act_and_mul_kernel":
            role = "shared_act"
        elif n.startswith("triton_poi_fused_clamp"):
            role = "moe_act"
        elif n == "_mtp_weighted_reduce_kernel" or (n == "vectorized_elementwise_kernel" and g == (24, 1, 1)):
            role = "moe_combine"
        elif n.startswith(("_fused_q_kv_rmsnorm", "rms_norm")):
            role = "norm"
        elif n.startswith(("_save_partial_states", "_fused_kv_compress")):
            role = "compressor"
        elif n.startswith(("_dequant_gather", "quantize_and_insert", "_sm70_qnorm_rope", "_sm70_inverse_rope",
                           "_sink_softmax", "_compute_", "_fill_short", "_build_c128a", "_prepare_dflash",
                           "elementwise_kernel_with_index", "indexSelect", "_compressed_slot")):
            role = "attn_aux"
        else:
            role = "elementwise"
        # der Schwanz der letzten Zielschicht (reduce, vec, post_dot, hc_head) bleibt bis zum Kopf bei der Schicht
        out.append((role, layer_local if region in ("layer", "draft") else None, region))
    return out


# ---------------------------------------------------------------- Auswertung
def layer_windows(lst):
    """Wandzeit-Fenster einer Schicht (us): Compressor-/Indexer-GEMMs, Attention-Kern, FFN-Vorspann, MoE-Block."""
    ks = [(r, k) for r, k in lst]
    def first(role, after=0):
        return next((i for i, (r, _) in enumerate(ks) if r == role and i >= after), None)
    def last(role):
        idx = [i for i, (r, _) in enumerate(ks) if r == role]
        return idx[-1] if idx else None
    w = {}
    a, nm = last("wq_a+wkv"), first("norm")
    if a is not None and nm is not None and nm > a:
        w["compress_fenster"] = (ks[nm][1]["s"] - ks[a][1]["e"]) / 1e3
    q, wa = last("wq_b"), first("wo_a")
    if q is not None and wa is not None and wa > q:
        w["attention_kern"] = (ks[wa][1]["s"] - ks[q][1]["e"]) / 1e3
        inner = ks[q + 1:wa]
        covered = 0
        last_end = ks[q][1]["e"]
        for _, k in inner:
            covered += max(0, k["e"] - max(k["s"], last_end))
            last_end = max(last_end, k["e"])
        w["attention_kern_leer"] = (ks[wa][1]["s"] - ks[q][1]["e"] - covered) / 1e3
        w["attention_kern_kerne"] = float(len(inner))
        tiny = [k for _, k in inner if (k["e"] - k["s"]) / 1e3 < 6.0]
        w["attention_kern_kleine_kerne"] = float(len(tiny))
        w["attention_kern_kleine_us"] = sum((k["e"] - k["s"]) / 1e3 for k in tiny)
    wb, mu = last("wo_b"), first("moe_up")
    if wb is not None and mu is not None and mu > wb:
        w["ffn_vorspann"] = (ks[mu][1]["s"] - ks[wb][1]["e"]) / 1e3
    md = last("moe_down")
    if mu is not None and md is not None:
        w["moe_block"] = (ks[md][1]["e"] - ks[mu][1]["s"]) / 1e3
    return w


def analyse(path):
    K, _ = load(path)
    P = {d: decode_passes(K[d]) for d in range(5)}
    n = min(len(P[d]) for d in range(5))
    R = [[P[d][r] for d in range(5)] for r in range(n)]
    res = {"rounds": n, "n_passes": {d: len(P[d]) for d in range(5)}}
    per = [(R[r + 1][0][0]["s"] - R[r][0][0]["s"]) / 1e6 for r in range(n - 1)]
    res["round_ms"], res["round_stats"] = st.mean(per), stats(per)
    hand = collections.defaultdict(list)
    for r in range(n - 1):
        for d in range(4):
            hand[f"S{d}->S{d + 1}"].append((R[r][d + 1][0]["s"] - R[r][d][-1]["e"]) / 1e6)
        hand["S4->S0"].append((R[r + 1][0][0]["s"] - R[r][4][-1]["e"]) / 1e6)
    res["handoff"] = {k: stats(v) for k, v in hand.items()}
    res["handoff_ms"] = sum(st.mean(v) for v in hand.values())
    res["stage"] = {d: {"dur": st.mean(dur(p) for p in P[d]), "busy": st.mean(busy(p) for p in P[d])} for d in range(5)}
    pre = []
    for p in P[0]:
        i = next(i for i, k in enumerate(p) if k["n"].startswith(("_sm70_mhc_prenorm_staging", "sm70_glm_mhc_pre_norm")))
        pre.append((p[i]["s"] - p[0]["s"]) / 1e6)
    res["prep_s0"] = stats(pre)
    res["prep_pass"] = {}
    for d in range(1, 5):
        ps = [s for s in segments(K[d]) if len(s) < 60 and s[0]["n"] in ("_post_update_kernel", "_apply_write_kernel")
              and not any(is_moe(k) for k in s)]
        res["prep_pass"][d] = (len(ps), st.mean(dur(p) for p in ps), st.mean(len(p) for p in ps)) if ps else None
    res["nccl"] = {}
    for d in range(5):
        tot, cnt = [], []
        for p in P[d]:
            ks = [k for k in K[d] if is_nccl(k) and k["s"] >= p[0]["s"] and k["e"] <= p[-1]["e"]]
            tot.append(sum(k["e"] - k["s"] for k in ks) / 1e3)
            cnt.append(len(ks))
        res["nccl"][d] = (st.mean(cnt), st.mean(tot))
    role_tab, layer_tab, region_tab, win_tab, moe_inst = {}, {}, {}, {}, {}
    for d in range(5):
        acc = collections.defaultdict(list)
        reg = collections.defaultdict(float)
        lt = collections.defaultdict(list)
        wins = collections.defaultdict(lambda: collections.defaultdict(list))
        mo = collections.defaultdict(lambda: {"up": [], "down": []})
        for p in P[d]:
            cl = classify(d, p)
            per_role = collections.defaultdict(list)
            per_layer = collections.defaultdict(list)
            reg_t = collections.defaultdict(float)
            for k, (role, L, rg) in zip(p, cl):
                du = (k["e"] - k["s"]) / 1e3
                per_role[role].append(du)
                reg_t[rg] += du
                if L is not None:
                    per_layer[L].append((role, k))
                if role in ("moe_up", "moe_down"):
                    mo[L if L is not None else -1]["up" if role == "moe_up" else "down"].append(du)
            for r, v in per_role.items():
                acc[r].append(v)
            for rg, v in reg_t.items():
                reg[rg] += v
            for L, lst in per_layer.items():
                gl = FIRST_LAYER[d] + L if L < TARGET_LAYERS[d] else 43 + (L - TARGET_LAYERS[d])
                typ = (layer_ratio(gl), gl >= 43)
                lt[typ].append(((sum((k["e"] - k["s"]) / 1e3 for _, k in lst)), (lst[-1][1]["e"] - lst[0][1]["s"]) / 1e3))
                for name, val in layer_windows(lst).items():
                    wins[typ][name].append(val)
        N = len(P[d])
        role_tab[d] = {r: {"per_round": sum(sum(v) for v in vs) / N, "n": sum(len(v) for v in vs) / N,
                               "mean": st.mean([x for v in vs for x in v]),
                               "mn": min(x for v in vs for x in v), "mx": max(x for v in vs for x in v)}
                       for r, vs in acc.items()}
        region_tab[d] = {k: v / N for k, v in reg.items()}
        layer_tab[d] = {f"ratio{t[0]}{'-draft' if t[1] else ''}": {
            "n_per_round": len(v) / N, "busy_us": st.mean(x[0] for x in v), "span_us": st.mean(x[1] for x in v)}
            for t, v in lt.items()}
        win_tab[d] = {f"ratio{t[0]}{'-draft' if t[1] else ''}": {k: st.mean(v) for k, v in w.items()} for t, w in wins.items()}
        moe_inst[d] = mo
    res.update(role_tab=role_tab, layer_tab=layer_tab, region_tab=region_tab, win_tab=win_tab, moe_inst=moe_inst)
    res["mhc_pre"] = {}
    for d in range(5):
        v = [(k["e"] - k["s"]) / 1e3 for p in P[d] for k in p if SIDE_PRE.match(k["n"])]
        res["mhc_pre"][d] = (st.mean(v), len(v) / len(P[d]))
    res["gaps"] = {}
    for d in range(5):
        cls = {"<1,5 us": [0, 0.0], "1,5-3 us": [0, 0.0], "3-10 us": [0, 0.0], ">10 us": [0, 0.0]}
        nk = tiny_n = 0
        tiny_us = 0.0
        for p in P[d]:
            last_end = p[0]["e"]
            for k in p[1:]:
                gap = (k["s"] - last_end) / 1e3
                if gap > 0:
                    key = "<1,5 us" if gap < 1.5 else "1,5-3 us" if gap < 3 else "3-10 us" if gap < 10 else ">10 us"
                    cls[key][0] += 1
                    cls[key][1] += gap
                last_end = max(last_end, k["e"])
            nk += len(p)
            for k in p:
                if (k["e"] - k["s"]) / 1e3 < 6.0:
                    tiny_n += 1
                    tiny_us += (k["e"] - k["s"]) / 1e3
        N = len(P[d])
        res["gaps"][d] = {"classes": {k: (v[0] / N, v[1] / N) for k, v in cls.items()}, "kernels": nk / N,
                              "tiny_n": tiny_n / N, "tiny_us": tiny_us / N}
    # Stufe 4 Abschnitte
    pieces = collections.defaultdict(list)
    for p in P[4]:
        cl = classify(4, p)
        idx_h = [i for i, (r, _, _) in enumerate(cl) if r == "lm_head"]
        idx_moe = [i for i, k in enumerate(p) if is_moe(k)]
        first_layer = next(i for i, k in enumerate(p) if SIDE_PRE.match(k["n"]))
        last_target_moe = idx_moe[2 * TARGET_LAYERS[4] - 1]
        pieces["Vorbereitung bis erste Schicht"].append((p[first_layer]["s"] - p[0]["s"]) / 1e6)
        pieces["Zielschichten 34-42 (erste mHC bis letzter MoE-down)"].append((p[last_target_moe]["e"] - p[first_layer]["s"]) / 1e6)
        pieces["Schwanz bis Kopf 1"].append((p[idx_h[0]]["s"] - p[last_target_moe]["e"]) / 1e6)
        pieces["Kopf 1 (Verify, BF16 129280x4096)"].append((p[idx_h[0]]["e"] - p[idx_h[0]]["s"]) / 1e6)
        first_draft_pre = next(i for i in range(idx_h[0], len(p)) if SIDE_PRE.match(p[i]["n"]))
        pieces["Kopf 1 Ende bis erste Entwurfs-mHC (Abnahme, main_proj, Kontext-Projektionen)"].append(
            (p[first_draft_pre]["s"] - p[idx_h[0]]["e"]) / 1e6)
        pieces["Entwurfsschichten (erste Entwurfs-mHC bis Kopf 2)"].append((p[idx_h[1]]["s"] - p[first_draft_pre]["s"]) / 1e6)
        pieces["Kopf 2 (Entwurf)"].append((p[idx_h[1]]["e"] - p[idx_h[1]]["s"]) / 1e6)
        pieces["Markov-Kopf 5 Schritte + Rest bis Durchlaufende"].append((p[-1]["e"] - p[idx_h[1]]["e"]) / 1e6)
    res["s4_pieces"] = {k: (st.mean(v), min(v), max(v)) for k, v in pieces.items()}
    allr = collections.defaultdict(float)
    for d in range(5):
        for r, v in role_tab[d].items():
            allr[r] += v["per_round"] / 1e3
    res["roles_ms"] = dict(allr)
    return res


def hebel(res):
    """Rechnung der Hebel (ms je Runde; Annahmen stehen je Zeile). Nicht addierbar, wo Fenster sich ueberlappen."""
    rt, wt, lt = res["role_tab"], res["win_tab"], res["layer_tab"]
    rtx, v100 = (0, 4), (1, 2, 3)
    out = []

    def save_role(role, devs, target):
        return sum(rt[d][role]["n"] * max(0.0, rt[d][role]["mean"] - target(d)) for d in devs) / 1e3

    # 1. wo_a Turing: FP8-Gruppenkern statt cuBLAS-FP16 auf entpacktem Gewicht (Ziel: Zeit von wo_b mit gleichen Bytes)
    out.append(("wo_a RTX als FP8-Gruppenkern",
                save_role("wo_a", rtx, lambda d: rt[d]["wo_b"]["mean"] + 10),
                save_role("wo_a", rtx, lambda d: rt[d]["wo_b"]["mean"]),
                save_role("wo_a", rtx, lambda d: BYTES["wo_a_fp8"] / (0.85 * PEAK[d] * 1e3))))
    # 2. wo_a V100: TurboMind-Gruppenkern auf wq_b-Niveau
    out.append(("wo_a V100 (TurboMind) auf wq_b-Niveau",
                save_role("wo_a", v100, lambda d: rt[d]["wo_b"]["mean"]),
                save_role("wo_a", v100, lambda d: rt[d]["wq_b"]["mean"]),
                save_role("wo_a", v100, lambda d: BYTES["wo_a_fp8"] / (0.85 * PEAK[d] * 1e3))))
    # 3. mHC pre_norm auf mehr CTAs (nur die pre_norm-Kerne; Zeit je Aufruf aus Rollen mhc, Anteil pre_norm gemessen unten)
    pre = res.get("mhc_pre", {})
    if pre:
        def mh(target):
            return sum(pre[d][1] * max(0.0, pre[d][0] - target) for d in range(5)) / 1e3
        out.append(("mHC pre_norm auf mehr CTAs (Ziel 11/8/5 us)", mh(11), mh(8), mh(5)))
    # 4. Router-Kette (Gate-GEMM + Reduce + Top-k + Gruppenrouting) in einen Kern: Ziel je Schicht 18/12/8 us
    def router(target):
        s = 0.0
        for d in range(5):
            layers = TARGET_LAYERS[d] + DRAFT_LAYERS.get(d, 0)
            per_layer = (rt[d]["router_gate"]["per_round"] + rt[d]["router_topk"]["per_round"]) / layers
            s += layers * max(0.0, per_layer - target)
        return s / 1e3
    out.append(("Router-Kette in einen Kern (Ziel 18/12/8 us je Schicht)", router(18), router(12), router(8)))
    # 5. Compressor-/Indexer-GEMM-Fenster auf Bytezeit (Ziel: 2x / 1,5x / 1x der 85-%-Zeit)
    bytes_win = {"ratio4": 21.49e6, "ratio128": 8.39e6}
    def comp(f):
        s = 0.0
        for d in range(5):
            for typ, b in bytes_win.items():
                w = wt[d].get(typ, {}).get("compress_fenster")
                if w is None:
                    continue
                n = lt[d][typ]["n_per_round"]
                s += n * max(0.0, w - f * b / (0.85 * PEAK[d] * 1e3))
        return s / 1e3
    out.append(("Compressor-/Indexer-GEMMs auf Bytezeit (Ziel 2x/1,5x/1x)", comp(2.0), comp(1.5), comp(1.0)))
    # 6. Attention-Kern: kleine Kerne (<6 us) und Leerlauf im Fenster auf ein Ziel je Schicht (40/25/15 us)
    def attn(target):
        s = 0.0
        for d in range(5):
            for typ, w in wt[d].items():
                if "attention_kern_kleine_us" not in w:
                    continue
                n = lt[d][typ]["n_per_round"]
                s += n * max(0.0, w["attention_kern_kleine_us"] + w["attention_kern_leer"] - target)
        return s / 1e3
    out.append(("Attention-Kern: kleine Kerne + Leerlauf fusionieren (Ziel 40/25/15 us je Schicht)", attn(40), attn(25), attn(15)))
    # 7. Entwurfskopf (Kopf 2) und Markov-Kopf nur fuer den Entwurf: FP8 bzw. Teilvokabular 98304
    h2 = rt[4]["lm_head"]["mean"] / 1e3
    mk = rt[4]["markov_head"]["per_round"] / 1e3
    out.append(("Entwurfskopf (Kopf 2) Teilvokabular 98304 / FP8 / FP8 + Teilvokabular", h2 * (1 - 98304 / 129280),
                h2 * 0.5, h2 * (1 - 0.5 * 98304 / 129280)))
    out.append(("Markov-Kopf FP8 (nur Gemv) / + Reduce-Kern fuer Argmax fusionieren / Teilvokabular dazu",
                mk * 0.5 * 0.78, mk * 0.5 * 0.78 + 5 * 0.025, mk * 0.5 * 0.78 + 5 * 0.025 + mk * 0.78 * 0.25 * 0.5))
    return out


def report(res, tokens):
    n = res["rounds"]
    print("Decode-Runden je Stufe:", res["n_passes"], "-> ausgerichtet", n)
    print(f"\n== Rundenzeit (Beginn Stufe 0 bis Beginn der naechsten Runde), ms: {res['round_stats']}")
    tpr = (tokens - 1) / n
    print(f"   Tokens je Runde {tokens - 1}/{n} = {tpr:.2f}; Rate unter nsys {tpr / res['round_ms'] * 1e3:.1f} tok/s")
    for k, v in res["handoff"].items():
        print(f"  Uebergabe {k}: {v}")
    print(f"  Summe Uebergaben {res['handoff_ms']:.2f} ms")
    print("\n  Stufe     Durchlauf ms  belegt ms  Luecken ms  Anteil Runde %  Leerlauf ms/Runde  Auslastung %")
    S = 0
    for d in range(5):
        du, bu = res["stage"][d]["dur"], res["stage"][d]["busy"]
        S += du
        print(f"  {NAME[d]:8s} {du:12.2f} {bu:10.2f} {du - bu:11.2f} {100 * du / res['round_ms']:14.1f} "
              f"{res['round_ms'] - bu:17.2f} {100 * bu / res['round_ms']:12.1f}")
    print(f"  Summe Durchlaeufe {S:.2f} + Uebergaben {res['handoff_ms']:.2f} = {S + res['handoff_ms']:.2f} (Rundenzeit {res['round_ms']:.2f})")
    print("\n  Stufe 0 Vorbereitung bis erster Schichtkern ms:", res["prep_s0"])
    for d, v in res["prep_pass"].items():
        if v:
            print(f"  {NAME[d]} Vorbereitungsdurchlauf (parallel zur Vorstufe, nicht kritisch): {v[0]} Stueck, {v[1]:.2f} ms, {v[2]:.0f} Kerne")
    for d, (c, t) in res["nccl"].items():
        print(f"  {NAME[d]} NCCL im Durchlauf: {c:.1f} Kerne, {t:.1f} us")
    print("\n== Rollen je Stufe (us je Runde; Anzahl je Runde; Mittel [min..max] je Aufruf)")
    for d in range(5):
        rt = res["role_tab"][d]
        tot = sum(r["per_round"] for r in rt.values())
        print(f"\n--- {NAME[d]} (Durchlauf {res['stage'][d]['dur']:.2f} ms, Kernsumme {tot / 1e3:.2f} ms)")
        for r, v in sorted(rt.items(), key=lambda x: -x[1]["per_round"]):
            if v["per_round"] < 15:
                continue
            print(f"  {r:16s} {v['per_round']:8.0f} us/Runde ({100 * v['per_round'] / tot:4.1f} %)  n {v['n']:5.1f}  "
                  f"Mittel {v['mean']:7.1f} [{v['mn']:.1f}..{v['mx']:.1f}] us")
        print("  Regionen (us/Runde):", {k: round(v) for k, v in res["region_tab"][d].items()})
        print("  Schichttypen (Instanzen/Runde, Kernsumme us, Spanne us):",
              {k: (round(v["n_per_round"], 1), round(v["busy_us"]), round(v["span_us"])) for k, v in res["layer_tab"][d].items()})
        print("  Fenster (us je Schicht):", {k: {a: round(b) for a, b in w.items()} for k, w in res["win_tab"][d].items()})
        g = res["gaps"][d]
        print(f"  Kerne je Runde {g['kernels']:.0f}; davon unter 6 us: {g['tiny_n']:.0f} Kerne = {g['tiny_us']:.0f} us; Leerlauf zwischen Kernen (Anzahl, us je Runde):",
              {k: (round(v[0]), round(v[1])) for k, v in g["classes"].items()})
    print("\n== Bandbreite je Rolle (Bytes aus Headern; GB/s = Byte/us/1000; % der nominellen Spitze)")
    for d in range(5):
        fp8 = d in (1, 2, 3)
        for role in ("wq_a+wkv", "wq_b", "wo_a", "wo_b", "shared_gu", "shared_dn", "indexer_wq_b", "main_proj", "lm_head"):
            v = res["role_tab"][d].get(role)
            byt = BYTES["wo_a_fp8" if fp8 else "wo_a_fp16"] if role == "wo_a" else BYTES[role]
            if not v:
                continue
            gb = byt / (v["mean"] * 1e3)
            ideal = byt / (0.85 * PEAK[d] * 1e3)
            print(f"  {NAME[d]:8s} {role:13s} {v['mean']:7.1f} us {byt / MB:8.2f} MB {gb:6.0f} GB/s = {100 * gb / PEAK[d]:5.1f} % ({100 * gb / PEAK_DRIVER[d]:5.1f} % Treiberwert)"
                  f"  bei 85 %: {ideal:6.1f} us  rest je Aufruf {max(0, v['mean'] - ideal):5.1f} us x n/Runde {v['n']:.0f}"
                  f" = {max(0, v['mean'] - ideal) * v["n"]:6.0f} us/Runde")
    print("\n== MoE-Kerne je Schicht (us, Mittel [min..max]): up / down")
    import numpy as np
    for d in range(5):
        print(f"  {NAME[d]}")
        ua, da = [], []
        for L in sorted(res["moe_inst"][d]):
            u, dn = res["moe_inst"][d][L]["up"], res["moe_inst"][d][L]["down"]
            gl = FIRST_LAYER[d] + L if L < TARGET_LAYERS[d] else 43 + L - TARGET_LAYERS[d]
            print(f"    L{gl:2d} up {st.mean(u):6.1f} [{min(u):6.1f}..{max(u):6.1f}] sd {st.pstdev(u):5.1f} | "
                  f"down {st.mean(dn):6.1f} [{min(dn):6.1f}..{max(dn):6.1f}] | up/down {st.mean(u) / st.mean(dn):.2f}")
            ua += u
            da += dn
        up, dn = np.array(ua), np.array(da)
        b, a0 = np.polyfit(dn, up, 1)
        print(f"    ALLE up Mittel {up.mean():.1f} p10 {np.percentile(up, 10):.0f} p90 {np.percentile(up, 90):.0f} max {up.max():.0f}; "
              f"down Mittel {dn.mean():.1f} max {dn.max():.0f}; Anpassung up = {a0:.1f} + {b:.2f} * down")
        rate = {0: 594, 4: 594}.get(d, 742)  # Grenzraten des Kerns aus MOE_MESSUNG.md (RTX 592-597, V100 mit Ring 739-746 GB/s)
        print(f"    angenommene Grenzrate {rate} GB/s: gleichwertige Zahl verschiedener Experten (Mittelzeit) up "
              f"{up.mean() * rate * 1e3 / EXPERT_UP:.1f}, down {dn.mean() * rate * 1e3 / EXPERT_DOWN:.1f}")
        print(f"    Bandbreite bei Maximum und {36} verschiedenen Experten: up {100 * 36 * EXPERT_UP / (up.max() * 1e3) / PEAK[d]:.1f} % "
              f"down {100 * 36 * EXPERT_DOWN / (dn.max() * 1e3) / PEAK[d]:.1f} %; N bei 87 % und Mittelzeit: "
              f"up {0.87 * PEAK[d] * up.mean() * 1e3 / EXPERT_UP:.1f} down {0.87 * PEAK[d] * dn.mean() * 1e3 / EXPERT_DOWN:.1f}")
    print("\n== Stufe 4 (Wandzeit der Abschnitte, ms)")
    for k, (m, lo, hi) in res["s4_pieces"].items():
        print(f"  {k:100s} {m:7.3f} [{lo:.3f}..{hi:.3f}]")
    print("\n== Hebelrechnung (ms je Runde, weniger ist besser; vorsichtig / Punkt / optimistisch; nicht addierbar bei Ueberlappung)")
    for name, lo, mid, hi in hebel(res):
        print(f"  {name:100s} {lo:6.2f} / {mid:6.2f} / {hi:6.2f}")
    print("  mHC pre_norm Mittel (us) und Aufrufe je Runde:", {NAME[d]: (round(v[0], 1), round(v[1])) for d, v in res["mhc_pre"].items()})
    print("\n== Gewichte-streamende Rollen gegen Rest (ms je Runde, Kernsummen)")
    stream = {"moe_up", "moe_down", "wq_a+wkv", "wq_b", "wo_a", "wo_b", "shared_gu", "shared_dn", "lm_head",
              "markov_head", "main_proj", "indexer_wq_b", "draft_ctx_proj"}
    s_ms = sum(v for r, v in res["roles_ms"].items() if r in stream)
    r_ms = sum(v for r, v in res["roles_ms"].items() if r not in stream)
    gap_ms = sum(res["stage"][d]["dur"] - res["stage"][d]["busy"] for d in range(5))
    print(f"  streamend {s_ms:.2f} ms ({100 * s_ms / res['round_ms']:.1f} %), uebrige Kerne {r_ms:.2f} ms "
          f"({100 * r_ms / res['round_ms']:.1f} %), Luecken im Durchlauf {gap_ms:.2f} ms ({100 * gap_ms / res['round_ms']:.1f} %), "
          f"Uebergaben {res['handoff_ms']:.2f} ms ({100 * res['handoff_ms'] / res['round_ms']:.1f} %)")
    print("\n== Rollen ueber alle Stufen (ms je Runde, Anteil an Rundenzeit)")
    for r, v in sorted(res["roles_ms"].items(), key=lambda x: -x[1]):
        print(f"  {r:16s} {v:7.3f} ms  {100 * v / res['round_ms']:5.1f} %")


def diff(a, b):
    print("\n== Unterschied Capture B gegen Capture A (us je Runde, Summe der Kernzeiten je Rolle und Stufe)")
    print(f"  Rundenzeit {a['round_ms']:.2f} -> {b['round_ms']:.2f} ms ({b['round_ms'] - a['round_ms']:+.2f})")
    print("  Stufe Durchlauf ms:", {NAME[d]: f"{a['stage'][d]['dur']:.2f}->{b['stage'][d]['dur']:.2f}" for d in range(5)})
    print("  Uebergaben ms:", f"{a['handoff_ms']:.2f}->{b['handoff_ms']:.2f}")
    rows = collections.defaultdict(lambda: [0.0, 0.0])
    for d in range(5):
        for r in set(a["role_tab"][d]) | set(b["role_tab"][d]):
            rows[r][0] += a["role_tab"][d].get(r, {"per_round": 0})["per_round"]
            rows[r][1] += b["role_tab"][d].get(r, {"per_round": 0})["per_round"]
    for r, (x, y) in sorted(rows.items(), key=lambda kv: -abs(kv[1][1] - kv[1][0])):
        if abs(y - x) >= 20:
            print(f"  {r:16s} {x:8.0f} -> {y:8.0f} us ({y - x:+7.0f})")
    print("  Fenster je Schicht (us), ratio4 / ratio128 / ratio0:")
    for d in range(5):
        for typ in ("ratio4", "ratio128", "ratio0"):
            wa, wb = a["win_tab"][d].get(typ), b["win_tab"][d].get(typ)
            if wa and wb:
                print(f"    {NAME[d]} {typ}: " + ", ".join(f"{k} {wa[k]:.0f}->{wb[k]:.0f}" for k in wa if k in wb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--vs", help="zweite Capture-Datei fuer den Vergleich (B gegen A)")
    ap.add_argument("--tokens", type=int, default=128, help="Ausgabetoken der Capture (Log: 128)")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = analyse(a.sqlite)
    report(res, a.tokens)
    if a.vs:
        res_b = analyse(a.vs)
        diff(res, res_b)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({k: v for k, v in res.items() if k != "moe_inst"}, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
