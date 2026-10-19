"""Kernelzeit je PP-Stufe nach Kernel-Familien, auf die Schichten der Stufe umgelegt.

Aufruf: python nsys_stage_families.py report.sqlite [weitere.sqlite ...] --layers 13,7,7,7,11
        [--cards RTX,V100,V100,V100,RTX] [--window S] [--top N]

Die Berichte (`nsys export -t sqlite`) dürfen ein oder mehrere Geräte enthalten. Stufen sind die `deviceId`s
aufsteigend über alle Berichte; bei PP mit einem Prozess je Stufe und `--trace-fork-before-exec` ist das die
sichtbare CUDA-Reihenfolge, also die Stufenreihenfolge. Die Kartennamen kommen aus `--cards`, weil die GPU-Tabelle
des Exports Systemnummern führt, nicht die Gerätenummern der Kernel.

Ausgabe je Stufe und Familie: ms gesamt und ms je Schicht der Stufe. Bei gleich schweren Schichten (GLM: MoE-
Schichten) zeigt die Spalte ms/Schicht direkt, wo eine Karte je Schicht langsamer ist.
"""

import argparse
import re
import sqlite3
from collections import defaultdict

# Erste passende Familie gewinnt; Reihenfolge von speziell nach allgemein.
FAMILIES = [
    # Includes waiting: PP broadcasts and SendRecv spin until the peer arrives.
    ("nccl", r"nccl|sendrecv|allreduce|allgather|broadcast"),
    ("csf_auspacken", r"_csf32_"),
    ("skinny_moe", r"skinny"),
    ("qpn8_fp8", r"qpn8"),
    (
        "kda_fla",
        r"kda|chunk_gla|recompute_w_u|delta_h|chunk_gated|fused_recurrent|kkt|gate_cumsum",
    ),
    ("indexer_kpool", r"indexer|kpool|topk|hadamard"),
    ("sparse_mla", r"glm5|sparse|mla"),
    ("mhc", r"mhc|hc_"),
    ("conv1d", r"conv1d"),
    # cuBLAS GEMV kernels show up as "Kernel2".
    ("gemm", r"gemm|gemv|kernel2|cutlass|s884|s1688|h884|h1688"),
    ("norm_act", r"norm|silu|act_and_mul"),
]


def family(name: str) -> str:
    lowered = name.lower()
    for fam, pattern in FAMILIES:
        if re.search(pattern, lowered):
            return fam
    return "sonstige"


def load(path: str, window_s: float | None) -> dict[int, dict[str, float]]:
    """Kernel-ms je deviceId und Kernelname."""
    db = sqlite3.connect(path)
    cols = {r[1] for r in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    name_col = "shortName" if "shortName" in cols else "demangledName"
    t_max = db.execute("SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    t_min = t_max - int(window_s * 1e9) if window_s else 0
    per_device: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for device, kname, ms in db.execute(
        f"""SELECT k.deviceId, s.value, SUM(k.end - k.start) / 1e6
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.{name_col}
            WHERE k.start >= {t_min} GROUP BY k.deviceId, s.value"""
    ):
        per_device[device][kname] += ms
    return per_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    parser.add_argument(
        "--layers", required=True, help="Schichten je Stufe, z. B. 13,7,7,7,11"
    )
    parser.add_argument(
        "--cards",
        default=None,
        help="Kartenname je Stufe, z. B. RTX,V100,V100,V100,RTX",
    )
    parser.add_argument(
        "--window", type=float, default=None, help="nur die letzten S Sekunden"
    )
    parser.add_argument(
        "--top", type=int, default=6, help="größte Einzelkernel je Stufe"
    )
    args = parser.parse_args()

    kernels: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for path in args.reports:
        for device, per_kernel in load(path, args.window).items():
            for name, ms in per_kernel.items():
                kernels[device][name] += ms
    devices = sorted(kernels)
    layers = [int(x) for x in args.layers.split(",")]
    cards = args.cards.split(",") if args.cards else [f"dev{d}" for d in devices]
    if len(layers) != len(devices) or len(cards) != len(devices):
        parser.error(
            f"{len(devices)} Geräte gefunden; --layers/--cards brauchen je einen Eintrag"
        )

    families = [defaultdict(float) for _ in devices]
    for i, device in enumerate(devices):
        for name, ms in kernels[device].items():
            families[i][family(name)] += ms
    order = sorted(
        {f for fams in families for f in fams},
        key=lambda f: -sum(x.get(f, 0.0) for x in families),
    )
    print(
        "Familie".ljust(16)
        + "".join(f"| S{i} {card:<6} ms  ms/Schicht " for i, card in enumerate(cards))
    )
    for fam in order + ["Summe"]:
        row = fam.ljust(16)
        for fams, n in zip(families, layers):
            ms = sum(fams.values()) if fam == "Summe" else fams.get(fam, 0.0)
            row += f"| {ms:10.1f} {ms / n:9.2f}  "
        print(row)
    for i, device in enumerate(devices):
        top = sorted(kernels[device].items(), key=lambda kv: -kv[1])[: args.top]
        print(
            f"\nS{i} {cards[i]} (deviceId {device}): "
            + "; ".join(f"{n[:60]} {ms:.1f} ms" for n, ms in top)
        )


if __name__ == "__main__":
    main()
