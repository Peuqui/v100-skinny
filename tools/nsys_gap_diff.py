"""Compare GPU idle gaps between consecutive kernels of two nsys capture sets.

Usage: python nsys_gap_diff.py <glob A> <glob B> [device] [top]
For one device, sums the idle time before each kernel (end of the previous kernel to start of this one),
grouped by the pair (previous kernel, this kernel), and lists the pairs whose gap total changed most (B - A).
Gaps hide host-side cost: Python between eager segments, launches, synchronizations.
"""
import glob
import sqlite3
import sys
from collections import defaultdict


def gaps(pattern: str, device: int) -> tuple[dict[tuple[str, str], list[float]], float]:
    pairs: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0])
    total = 0.0
    for path in sorted(glob.glob(pattern)):
        db = sqlite3.connect(path)
        rows = db.execute(
            "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id = k.shortName WHERE k.deviceId = ? ORDER BY k.start",
            (device,)).fetchall()
        previous_end, previous_name = None, "<start>"
        for start, end, name in rows:
            if previous_end is not None and start > previous_end:
                gap = (start - previous_end) / 1e6
                entry = pairs[(previous_name, name)]
                entry[0] += gap
                entry[1] += 1
                total += gap
            previous_end = max(end, previous_end or end)
            previous_name = name
    return {k: v for k, v in pairs.items()}, total


def main() -> None:
    device = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    top = int(sys.argv[4]) if len(sys.argv) > 4 else 15
    a, total_a = gaps(sys.argv[1], device)
    b, total_b = gaps(sys.argv[2], device)
    print(f"Gerät {device}: Lücken gesamt A {total_a:.1f} ms, B {total_b:.1f} ms, Differenz {total_b - total_a:+.1f} ms")
    rows = []
    for pair in set(a) | set(b):
        ms_a, n_a = a.get(pair, (0.0, 0))
        ms_b, n_b = b.get(pair, (0.0, 0))
        rows.append((ms_b - ms_a, ms_a, n_a, ms_b, n_b, pair))
    for delta, ms_a, n_a, ms_b, n_b, (prev, cur) in sorted(rows, key=lambda r: -abs(r[0]))[:top]:
        print(f"  {delta:+7.1f} ms  A {ms_a:7.1f}/{n_a:5d}  B {ms_b:7.1f}/{n_b:5d}  {prev[:45]} -> {cur[:45]}")


if __name__ == "__main__":
    main()
