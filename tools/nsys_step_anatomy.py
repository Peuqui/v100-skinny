"""Schritt-Anatomie eines Pipeline-Decodes aus einem nsys-SQLite-Export.

Aufruf: python nsys_step_anatomy.py report.sqlite [--step-kernel NAME] [--top N]

Mit --step-kernel beginnt ein Schritt mit jedem Aufruf dieses Kernels auf Gerät 0
statt mit dem großen Block (nötig, wenn der Vorwärtslauf der ersten Stufe in mehrere
Blöcke zerfällt, z. B. PP4-Decode: _combine_sampled_and_draft_tokens_kernel,
DSv4/DSpark: _prepare_uniform_decode_kernel).

NCCL-Kernel zählen nicht als Rechnen: ein Empfangskernel dreht auf der GPU, bis die
Vorstufe liefert, ist also Wartezeit. Je Gerät werden die Rechenkernel zu
Arbeitsblöcken verschmolzen. Ein Decode-Schritt beginnt mit dem großen Block auf
Gerät 0 (erste Pipeline-Stufe). Ausgegeben wird
je Gerät der Median über alle vollständigen Schritte: wann es relativ zum
Schrittbeginn anfängt und aufhört, wie lange es rechnet, dazu die Lücken an den
Nähten und die Umkehrzeit vom letzten Kernel des Schritts bis zum nächsten Beginn.
"""
import argparse
import sqlite3
import statistics
from collections import defaultdict

MERGE_GAP_NS = 300_000  # Kernel mit weniger Abstand gehören zu einem Block
STEP_BLOCK_MIN_NS = 4_000_000  # der Vorwärtslauf der ersten Stufe

parser = argparse.ArgumentParser()
parser.add_argument("report")
parser.add_argument("--step-kernel")
parser.add_argument("--top", type=int, default=15)
args = parser.parse_args()

db = sqlite3.connect(args.report)
rows = db.execute(
    """SELECT k.deviceId, k.start, k.end, s.value
       FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName
       ORDER BY k.deviceId, k.start"""
).fetchall()
kernels: dict[int, list[tuple[int, int]]] = {}
waits: dict[int, list[tuple[int, int]]] = {}
named: dict[int, list[tuple[int, int, str]]] = {}
for dev, start, end, name in rows:
    target = waits if "nccl" in name.lower() else kernels
    target.setdefault(dev, []).append((start, end))
    named.setdefault(dev, []).append((start, end, name))
devices = sorted(kernels)
print(f"Geräte: {devices}, Rechenkernel: {sum(map(len, kernels.values()))}, "
      f"NCCL-Kernel: {sum(map(len, waits.values()))}")


def blocks(intervals: list[tuple[int, int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in intervals:
        if merged and start - merged[-1][1] <= MERGE_GAP_NS:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


first = devices[0]
if args.step_kernel:
    starts = [s for s, _, name in named[first] if name == args.step_kernel]
else:
    starts = [b[0] for b in blocks(kernels[first]) if b[1] - b[0] >= STEP_BLOCK_MIN_NS]
steps = list(zip(starts[:-1], starts[1:]))
lengths = [(b - a) / 1e6 for a, b in steps]
typical = statistics.median(lengths)
steps = [s for s, length in zip(steps, lengths) if length < 1.5 * typical]
print(f"Schritte: {len(steps)}, Median {typical:.1f} ms "
      f"(min {min(lengths):.1f}, max {max(lengths):.1f})\n")

BLOCK_GAP_NS = 1_000_000  # innerhalb eines Vorwärtslaufs sind die Lücken kleiner
BLOCK_MIN_MS = 1.5  # kleinere Blöcke sind Metadaten-Vorbereitung und Zustandspflege

per_device: dict[int, list[list[tuple[float, float, float]]]] = {d: [] for d in devices}
waiting: dict[int, list[float]] = {d: [] for d in devices}
turnaround: list[float] = []
for begin, nxt in steps:
    last_any = begin
    for dev in devices:
        mine = [(s, e) for s, e in kernels[dev] if begin <= s < nxt]
        if not mine:
            continue
        merged: list[list[int]] = []
        busy: list[int] = []
        for start, end in mine:
            if merged and start - merged[-1][1] <= BLOCK_GAP_NS:
                merged[-1][1] = max(merged[-1][1], end)
                busy[-1] += end - start
            else:
                merged.append([start, end])
                busy.append(end - start)
        big = [
            ((b[0] - begin) / 1e6, (b[1] - begin) / 1e6, w / 1e6)
            for b, w in zip(merged, busy)
            if (b[1] - b[0]) / 1e6 >= BLOCK_MIN_MS
        ]
        per_device[dev].append(big)
        waiting[dev].append(
            sum(e - s for s, e in waits.get(dev, []) if begin <= s < nxt) / 1e6
        )
        last_any = max(last_any, max(e for _, e in mine))
    turnaround.append((nxt - last_any) / 1e6)

print(" Gerät | große Rechenblöcke je Schritt (Beginn–Ende, davon Kernelzeit) | NCCL-Warten")
total_busy = 0.0
for dev in devices:
    typical_count = statistics.mode(len(b) for b in per_device[dev])
    same = [b for b in per_device[dev] if len(b) == typical_count]
    parts = []
    for i in range(typical_count):
        start, end, work = (statistics.median(b[i][k] for b in same) for k in range(3))
        parts.append(f"{start:5.1f}–{end:5.1f} ms ({work:4.1f})")
        total_busy += work
    print(f"   {dev}   | {' | '.join(parts)} | {statistics.median(waiting[dev]):5.1f} ms")
print(f"\nUmkehrzeit (letzter Kernel des Schritts bis Beginn des nächsten): "
      f"Median {statistics.median(turnaround):.1f} ms")
print(f"Kernelzeit in den großen Blöcken aller Geräte: {total_busy:.1f} ms von {typical:.1f} ms je Schritt")

print(f"\nTeuerste Kernel je Gerät, Kernelzeit je Schritt (ms, Mittel über {len(steps)} Schritte)")
for dev in devices:
    totals: dict[str, float] = defaultdict(float)
    for begin, nxt in steps:
        for start, end, name in named[dev]:
            if begin <= start < nxt:
                totals[name] += (end - start) / 1e6
    busy = sum(v for k, v in totals.items() if "nccl" not in k.lower()) / len(steps)
    print(f"\n Gerät {dev}: Rechenkernel {busy:.1f} ms je Schritt")
    for name, total in sorted(totals.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"   {total / len(steps):6.2f}  {name[:90]}")
