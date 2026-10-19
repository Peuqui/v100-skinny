"""GPU-Zeit je Decode-Runde nach Kernelnamen aus nsys-SQLite-Exporten, ein oder zwei Profile.

Aufruf: python nsys_round_kernels.py a.sqlite [b.sqlite] [--device N] [--marker NAME] [--top N]

Eine Runde (Zielprüfung, Sampler, Entwurf) beginnt mit jedem Aufruf des Marker-Kernels auf dem
gewählten Gerät; der V2-Runner ruft _combine_sampled_and_draft_tokens_kernel genau einmal je Runde
auf, unabhängig vom Entwurfsverfahren. Gezählt werden nur vollständige Runden. Je Profil: Zahl der
Runden, Median der Rundendauer (Wand) und der Rechenzeit (Vereinigung der Kernel-Intervalle ohne
NCCL), dazu die NCCL-Zeit je Runde. Je Kernelname: Aufrufe und µs je Runde; mit zwei Profilen nach
dem Betrag des Unterschieds sortiert, so dass oben steht, wo B mehr oder weniger Zeit braucht als A.
"""

import argparse
import bisect
import sqlite3
import statistics
from collections import defaultdict

DEFAULT_MARKER = "_combine_sampled_and_draft_tokens_kernel"


def load(path: str, device: int | None, marker: str) -> dict:
    db = sqlite3.connect(path)
    rows = db.execute(
        """SELECT k.deviceId, k.start, k.end, s.value
           FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName
           ORDER BY k.start"""
    ).fetchall()
    devices = sorted({row[0] for row in rows})
    chosen = devices[0] if device is None else device
    kernels = [(start, end, name) for dev, start, end, name in rows if dev == chosen]
    marks = [start for start, _, name in kernels if name == marker]
    if len(marks) < 3:
        raise SystemExit(
            f"{path}: nur {len(marks)} Aufrufe von {marker} auf Gerät {chosen}"
        )
    rounds = len(marks) - 1
    per_name: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    walls, busy = [], [0.0] * rounds
    nccl = 0.0
    intervals: list[list[tuple[int, int]]] = [[] for _ in range(rounds)]
    for start, end, name in kernels:
        index = bisect.bisect_right(marks, start) - 1
        if not 0 <= index < rounds:
            continue
        per_name[name][0] += (end - start) / 1e3
        per_name[name][1] += 1
        if "nccl" in name.lower():
            nccl += (end - start) / 1e3
        else:
            intervals[index].append((start, end))
    for index in range(rounds):
        walls.append((marks[index + 1] - marks[index]) / 1e6)
        covered, last_end = 0, 0
        for start, end in sorted(intervals[index]):
            start = max(start, last_end)
            if end > start:
                covered += end - start
                last_end = end
        busy[index] = covered / 1e6
    return {
        "path": path,
        "device": chosen,
        "devices": devices,
        "rounds": rounds,
        "wall_ms": statistics.median(walls),
        "busy_ms": statistics.median(busy),
        "nccl_us": nccl / rounds,
        "per_name": {
            name: (total / rounds, calls / rounds)
            for name, (total, calls) in per_name.items()
        },
    }


def summary(label: str, profile: dict) -> None:
    print(
        f"{label}: {profile['path']}\n"
        f"   Gerät {profile['device']} von {profile['devices']}, {profile['rounds']} Runden, "
        f"Runde {profile['wall_ms']:.2f} ms (Median, Wand), Rechnen {profile['busy_ms']:.2f} ms "
        f"(Median, ohne NCCL), NCCL {profile['nccl_us'] / 1e3:.2f} ms je Runde"
    )


parser = argparse.ArgumentParser()
parser.add_argument("reports", nargs="+")
parser.add_argument("--device", type=int)
parser.add_argument("--marker", default=DEFAULT_MARKER)
parser.add_argument("--top", type=int, default=25)
args = parser.parse_args()
if len(args.reports) > 2:
    parser.error("höchstens zwei Profile")

profiles = [load(path, args.device, args.marker) for path in args.reports]
for label, profile in zip("AB", profiles):
    summary(label, profile)

if len(profiles) == 1:
    ranked = sorted(profiles[0]["per_name"].items(), key=lambda item: -item[1][0])
    print(f"\n{'µs/Runde':>10} {'Aufrufe':>8}  Kernel")
    for name, (micros, calls) in ranked[: args.top]:
        print(f"{micros:10.1f} {calls:8.2f}  {name[:110]}")
else:
    a, b = (profile["per_name"] for profile in profiles)
    names = set(a) | set(b)
    ranked = sorted(
        names, key=lambda n: -abs(b.get(n, (0.0, 0.0))[0] - a.get(n, (0.0, 0.0))[0])
    )
    print(
        f"\n{'A µs':>9} {'A Aufr.':>8} {'B µs':>9} {'B Aufr.':>8} {'B−A µs':>9}  Kernel (je Runde)"
    )
    for name in ranked[: args.top]:
        a_us, a_calls = a.get(name, (0.0, 0.0))
        b_us, b_calls = b.get(name, (0.0, 0.0))
        print(
            f"{a_us:9.1f} {a_calls:8.2f} {b_us:9.1f} {b_calls:8.2f} {b_us - a_us:9.1f}  {name[:100]}"
        )
    total_a = sum(us for us, _ in a.values())
    total_b = sum(us for us, _ in b.values())
    print(
        f"\nSumme aller Kernel je Runde: A {total_a / 1e3:.2f} ms, B {total_b / 1e3:.2f} ms"
    )
