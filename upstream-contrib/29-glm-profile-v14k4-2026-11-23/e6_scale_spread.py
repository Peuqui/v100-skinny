"""E6: Skalen-Spanne je K-Scheibe der GLM-5.3-Expertenprojektionen (nur Host, nur lesend, keine GPU).

Frage (MOE_KERN.md 7.2, E6): Passen die Skalen der Schichten, die heute wegen "Spanne > 31" den csf-Raster-Pfad
nehmen, mit einer Basis je K-Scheibe + 5 Bit Offset, oder braucht es Sechs-Bit-Planes?

Echter Code, nicht nachgebaut (Baum 1Cat-vLLM-v14, nur importiert):
  - csf-Decoder          vllm/model_executor/layers/quantization/utils/csf32_scales.py::decode_csf32
                         (derselbe Zerleger, dessen Ergebnis der GPU-Expander csf32_expand_kernel nachbildet)
  - Prepack/Raster       vllm/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py::qpn_prepack (:154)
  - Spanne heute         vllm/model_executor/layers/quantization/utils/s5_scales.py::scale_spread
  - Entscheidung S5/csf  skinny_sm70_moe.py:585-587 (_csf_to_s5: spread = scale_spread(...); fits = spread <= S5_MAX_SPREAD)
  - Kern-Aufteilung K    csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu:263-264 (qpn, V100) und :529-530 (m16n8k8, RTX):
                         G = K / 16, Gq = G / SPLITK, g0 = warp * Gq: jeder Warp eine zusammenhaengende Scheibe von Gq
                         Gruppen; SPLITK = grouped_splitk(K, ...) = 8 fuer K = 4096 (W13) und K = 2048 (W2) auf beiden Karten.

Die Spanne heute ist je Zeile (Tile-Position) max - min ueber alle K-Gruppen, danach das Maximum ueber alle Zeilen
und Experten der Projektion. Je Scheibe: dieselbe Rechnung ueber die Gq Gruppen eines Warps. Qpn-Prepack ist eine reine
Permutation der Spalten innerhalb einer 32er-Kachel; die Spanne je Zeile haengt davon nicht ab. Das Skript rechnet
direkt auf der dekodierten Matrix [N, K/16] (schnell) und belegt die Gleichheit mit dem echten qpn_prepack +
scale_spread an einigen Experten (beide Fragment-Ordnungen).

Aufruf (CPU, cwd ausserhalb des Baums):
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=<Baum> python e6_scale_spread.py --layers 35 36 ... --json aus.json
"""

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import (
    QpnFragmentOrder,
    grouped_splitk,
    qpn_prepack,
)
from vllm.model_executor.layers.quantization.utils.csf32_scales import decode_csf32
from vllm.model_executor.layers.quantization.utils.s5_scales import (
    S5_MAX_SPREAD,
    decode_s5,
    encode_s5,
    scale_spread,
)

CHECKPOINT = Path("/home/mp/models/GLM-5.3-Flash-NVFP4-MTP")
NUM_EXPERTS = 288
HIDDEN = 4096
INTER = 2048
TILE = 32
SIX_BIT_MAX = 63
# Projektion -> (Matrizen im Checkpoint, Zeilen N der Prepack-Matrix, K)
PROJECTIONS = {
    "W13": (("gate_proj", "up_proj"), 2 * INTER, HIDDEN),
    "W2": (("down_proj",), HIDDEN, INTER),
}
PART_NAMES = ("bases", "codes", "page_bases", "offsets", "exceptions")
CSF_PART_KEYS = {
    "bases": "csf32_bases",
    "codes": "csf32_codes",
    "page_bases": "csf32_page_bases",
    "offsets": "csf32_offsets",
    "exceptions": "csf32_exceptions",
}


def bits_for(spread: int) -> int:
    """Bits eines Offsets 0..spread."""
    return max(1, math.ceil(math.log2(spread + 1)))


class Checkpoint:
    def __init__(self) -> None:
        index = json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._open: dict[str, object] = {}

    def tensor(self, key: str) -> torch.Tensor:
        shard = self.weight_map[key]
        if shard not in self._open:
            self._open[shard] = safe_open(str(CHECKPOINT / shard), "pt")
        return self._open[shard].get_tensor(key)  # type: ignore[attr-defined]

    def matrix(self, layer: int, expert: int, name: str) -> tuple[torch.Tensor, dict[str, int]]:
        """Skalenbytes [rows, K/16] einer Expertenmatrix und die csf-Teilgroessen (Checkpoint-Bytes)."""
        prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{name}"
        parts = {p: self.tensor(f"{prefix}.weight_scale.{CSF_PART_KEYS[p]}") for p in PART_NAMES}
        rows = parts["bases"].numel()
        cols = parts["codes"].numel() * 2 // rows
        scales = decode_csf32(*(parts[p] for p in PART_NAMES), rows, cols)
        sizes = {
            "pages": parts["page_bases"].numel(),
            "blocks_plus_pages": parts["offsets"].numel(),
            "records": parts["exceptions"].numel(),
        }
        return scales, sizes


def slice_spread(matrix: torch.Tensor, pieces: int) -> torch.Tensor:
    """Spanne je (Zeile, Stueck) einer [N, G] u8-Matrix, die G in `pieces` gleiche Stuecke teilt: [N, pieces]."""
    rows, groups = matrix.shape
    m = matrix.view(rows, pieces, groups // pieces).to(torch.int16)
    return m.amax(dim=2) - m.amin(dim=2)


def verify_with_real_prepack(scales: torch.Tensor, rows: int, groups: int, pieces: int) -> None:
    """Gleichheit: Spanne aus qpn_prepack-Raster (echter Code) == direkte Rechnung, beide Fragment-Ordnungen."""
    direct_global = int(slice_spread(scales, 1).max())
    direct_slice = int(slice_spread(scales, pieces).max())
    codes = torch.zeros(rows, groups * 8, dtype=torch.uint8)
    for order in QpnFragmentOrder:
        _, raster = qpn_prepack(codes, scales, 16, order)
        assert raster is not None
        via_global = scale_spread(raster.view(1, -1), groups)
        per_row = raster.view(-1, pieces, groups // pieces, TILE).to(torch.int16)
        via_slice = int((per_row.amax(dim=2) - per_row.amin(dim=2)).max())
        if (via_global, via_slice) != (direct_global, direct_slice):
            raise AssertionError(
                f"{order.name}: Prepack ({via_global}, {via_slice}) != direkt ({direct_global}, {direct_slice})"
            )


def analyze_layer(layer: int, verify_experts: int) -> dict:
    torch.set_num_threads(2)
    ckpt = Checkpoint()
    result: dict = {"layer": layer, "projections": {}}
    for proj, (names, rows, k) in PROJECTIONS.items():
        groups = k // 16
        splitk = grouped_splitk(k, QpnFragmentOrder.M16N8K8)
        assert splitk == grouped_splitk(k, QpnFragmentOrder.M8N8K4)
        per_warp = groups // splitk
        granularities = {"Scheibe": splitk, "halbe": splitk * 2, "viertel": splitk * 4}
        glob = []  # je Experte: Spanne global
        sl: dict[str, list[int]] = {g: [] for g in granularities}
        warp_max = [0] * splitk
        rows_total = rows_over31_global = slices_total = slices_over31 = 0
        csf_records = 0
        for expert in range(NUM_EXPERTS):
            mats = []
            for name in names:
                scales, sizes = ckpt.matrix(layer, expert, name)
                mats.append(scales)
                csf_records += sizes["records"]
            matrix = torch.cat(mats, dim=0)
            assert matrix.shape == (rows, groups), matrix.shape
            row_spread = slice_spread(matrix, 1)[:, 0]
            glob.append(int(row_spread.max()))
            rows_total += rows
            rows_over31_global += int((row_spread > S5_MAX_SPREAD).sum())
            for label, pieces in granularities.items():
                spread = slice_spread(matrix, pieces)
                sl[label].append(int(spread.max()))
                if label == "Scheibe":
                    slices_total += spread.numel()
                    slices_over31 += int((spread > S5_MAX_SPREAD).sum())
                    for warp in range(splitk):
                        warp_max[warp] = max(warp_max[warp], int(spread[:, warp].max()))
            if expert < verify_experts:
                verify_with_real_prepack(matrix, rows, groups, splitk)
                if glob[-1] <= S5_MAX_SPREAD:  # Kontrolle: S5-Rundreise nur dort, wo S5 heute gilt
                    _, raster = qpn_prepack(torch.zeros(rows, k // 2, dtype=torch.uint8), matrix, 16, QpnFragmentOrder.M16N8K8)
                    assert raster is not None
                    flat = raster.view(1, -1)
                    assert torch.equal(decode_s5(encode_s5(flat, groups)).reshape(1, -1), flat)
        result["projections"][proj] = {
            "rows": rows,
            "k": k,
            "groups": groups,
            "splitk": splitk,
            "groups_per_warp": per_warp,
            "global_max": max(glob),
            "global_per_expert": glob,
            "slice_max": {g: max(v) for g, v in sl.items()},
            "slice_per_expert": sl["Scheibe"],
            "warp_max": warp_max,
            "experts_global_le31": sum(v <= S5_MAX_SPREAD for v in glob),
            "experts_global_le63": sum(v <= SIX_BIT_MAX for v in glob),
            "experts_slice_le31": {g: sum(v <= S5_MAX_SPREAD for v in vals) for g, vals in sl.items()},
            "rows_total": rows_total,
            "rows_over31_global": rows_over31_global,
            "slices_total": slices_total,
            "slices_over31": slices_over31,
            "csf_records_per_expert": csf_records / NUM_EXPERTS,
            "verified_experts": verify_experts,
        }
    return result


def vram_bytes(proj: str, records_per_expert: float) -> dict[str, float]:
    """Bytes je Experte einer Projektion (aus den Tensorformen der Kern-Vorgaben, s5_scales.py)."""
    _, rows, k = PROJECTIONS[proj]
    groups = k // 16
    tiles = rows // TILE
    s5 = tiles * (TILE + groups // 2 * TILE + groups * 4)
    six = tiles * (TILE + groups // 2 * TILE + groups * 8)  # zweites Hi-Wort je (Kachel, Gruppe)
    per_slice = tiles * (TILE * 8 + groups // 2 * TILE + groups * 4)  # 8 Basen statt einer
    raster = tiles * groups * TILE
    matrices = 2 if proj == "W13" else 1
    # csf auf der GPU (CsfExpertScales): bases u8, codes u8, page_bases i32, offsets i32, exceptions i32, i64-Start je Matrix.
    pages_each = -(-((rows // matrices) // 32 * (groups // 2)) // 256)
    blocks_each = (rows // matrices) // 32 * (groups // 2)
    csf = (
        rows
        + rows * groups // 2
        + matrices * (4 * pages_each + 4 * (blocks_each + pages_each) + 8)
        + 4 * records_per_expert
    )
    return {"s5": s5, "six_bit": six, "per_slice_base": per_slice, "raster": raster, "csf_gpu": csf}


def fmt_pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:.2f} ({a}/{b})"


def tables(results: list[dict]) -> str:
    out = []
    out.append("| Schicht | Proj. | max Spanne global (Bytes e4m3) | max Spanne je Scheibe (Warp, Gq Gruppen) | je halbe Scheibe | je Viertel | Experten global <= 31 % (Anzahl) | Experten Scheibe <= 31 % (Anzahl) | Zeilen global > 31 % (Anzahl) | Zeilen-Scheiben > 31 % (Anzahl) | 6 Bit global reicht (<= 63) | Bits global | Bits je Scheibe |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for res in sorted(results, key=lambda r: r["layer"]):
        for proj, p in res["projections"].items():
            n = NUM_EXPERTS
            six = "ja" if p["global_max"] <= SIX_BIT_MAX else "nein"
            out.append(
                f"| {res['layer']} | {proj} | {p['global_max']} | {p['slice_max']['Scheibe']} | {p['slice_max']['halbe']} | "
                f"{p['slice_max']['viertel']} | {fmt_pct(p['experts_global_le31'], n)} | "
                f"{fmt_pct(p['experts_slice_le31']['Scheibe'], n)} | {fmt_pct(p['rows_over31_global'], p['rows_total'])} | "
                f"{fmt_pct(p['slices_over31'], p['slices_total'])} | {six} | {bits_for(p['global_max'])} | {bits_for(p['slice_max']['Scheibe'])} |"
            )
    return "\n".join(out)


def vram_net(results: list[dict]) -> str:
    """MB der csf-Projektionen heute gegen Sechs-Bit-Planes bzw. Basis je Scheibe (288 Experten je Projektion)."""
    rows = []
    totals = {"csf": 0.0, "six": 0.0, "slice": 0.0}
    all_slice_fit = True
    for res in sorted(results, key=lambda r: r["layer"]):
        for proj, p in res["projections"].items():
            if p["global_max"] <= S5_MAX_SPREAD:
                continue
            v = vram_bytes(proj, p["csf_records_per_expert"])
            fits_slice = p["slice_max"]["Scheibe"] <= S5_MAX_SPREAD
            all_slice_fit &= fits_slice
            csf, six, per_slice = (v[k] * NUM_EXPERTS / 1e6 for k in ("csf_gpu", "six_bit", "per_slice_base"))
            totals["csf"] += csf
            totals["six"] += six
            totals["slice"] += per_slice
            rows.append(
                f"| {res['layer']} | {proj} | {csf:.1f} | {six:.1f} | {per_slice:.1f} | {'ja' if fits_slice else 'nein'} |"
            )
    raster = sum(vram_bytes(proj, 0.0)["raster"] for proj in PROJECTIONS) * NUM_EXPERTS / 1e6
    head = [
        "csf-Projektionen (globale Spanne > 31), MB je Projektion und Schicht (288 Experten):",
        "",
        "| Schicht | Proj. | csf-Planes heute | Sechs-Bit-Planes | Basis je Scheibe | Basis je Scheibe passt |",
        "|---|---|---|---|---|---|",
    ]
    tail = [
        f"| Summe | | {totals['csf']:.1f} | {totals['six']:.1f} | {totals['slice']:.1f} | |",
        "",
        f"Gemeinsames Raster je Karte (W13 + W2, 288 Experten, entfaellt, sobald keine Schicht der Karte mehr csf haelt): {raster:.1f} MB.",
        f"Netto Sechs-Bit: {totals['six'] - totals['csf'] - raster:+.1f} MB; Basis je Scheibe: {totals['slice'] - totals['csf'] - raster:+.1f} MB "
        f"(nur gueltig, wenn alle Projektionen passen: {'ja' if all_slice_fit else 'nein'}).",
    ]
    return "\n".join(head + rows + tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layers", type=int, nargs="+", default=[14, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--verify-experts", type=int, default=2, help="Experten je Projektion, die gegen den echten qpn_prepack geprueft werden")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if torch.cuda.is_initialized():
        sys.exit("CUDA-Kontext darf nicht angelegt sein")
    start = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.starmap(analyze_layer, [(layer, args.verify_experts) for layer in args.layers])
    print(tables(results))
    print()
    print("| Projektion | S5 (heute) | Sechs-Bit-Planes | Basis je Scheibe (8 Basen) | Raster (gemeinsam, je Experte) |")
    print("|---|---|---|---|---|")
    for proj in PROJECTIONS:
        v = vram_bytes(proj, 0.0)
        print(f"| {proj} | {v['s5']/1e6:.4f} | {v['six_bit']/1e6:.4f} | {v['per_slice_base']/1e6:.4f} | {v['raster']/1e6:.4f} |")
    print()
    print(vram_net(results))
    print(f"\nLaufzeit {time.time() - start:.0f} s")
    if args.json:
        args.json.write_text(json.dumps(results))


if __name__ == "__main__":
    main()
