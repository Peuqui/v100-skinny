"""Compare GPU kernel time per device and kernel name between two nsys capture sets.

Usage: python nsys_kernel_diff.py <glob A> <glob B> [top]
Each glob matches the SQLite exports of one capture (one file per process). Kernels are grouped by
device and short name; the diff lists the largest absolute changes (B - A) per device and overall.
"""
import glob
import sqlite3
import sys
from collections import defaultdict


def load(pattern: str) -> tuple[dict[tuple[int, str], tuple[float, int]], dict[int, float]]:
    kernels: dict[tuple[int, str], list[float]] = defaultdict(lambda: [0.0, 0])
    spans: dict[int, float] = {}
    for path in sorted(glob.glob(pattern)):
        db = sqlite3.connect(path)
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
            continue
        rows = db.execute(
            "SELECT k.deviceId, s.value, SUM(k.end - k.start), COUNT(*), MIN(k.start), MAX(k.end) "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName "
            "GROUP BY k.deviceId, s.value").fetchall()
        for dev, name, ns, count, start, end in rows:
            entry = kernels[(dev, name)]
            entry[0] += ns / 1e6
            entry[1] += count
        for dev, start, end in db.execute(
                "SELECT deviceId, MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY deviceId"):
            spans[dev] = (end - start) / 1e6
    return {k: (v[0], int(v[1])) for k, v in kernels.items()}, spans


def main() -> None:
    a, spans_a = load(sys.argv[1])
    b, spans_b = load(sys.argv[2])
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    devices = sorted({dev for dev, _ in a} | {dev for dev, _ in b})
    for dev in devices:
        total_a = sum(ms for (d, _), (ms, _) in a.items() if d == dev)
        total_b = sum(ms for (d, _), (ms, _) in b.items() if d == dev)
        print(f"\n=== Gerät {dev}: Kernelzeit A {total_a:.1f} ms, B {total_b:.1f} ms, Differenz {total_b - total_a:+.1f} ms"
              f" (Spanne A {spans_a.get(dev, 0):.0f} ms, B {spans_b.get(dev, 0):.0f} ms)")
        names = {n for d, n in a if d == dev} | {n for d, n in b if d == dev}
        diffs = []
        for name in names:
            ms_a, n_a = a.get((dev, name), (0.0, 0))
            ms_b, n_b = b.get((dev, name), (0.0, 0))
            diffs.append((ms_b - ms_a, ms_a, n_a, ms_b, n_b, name))
        for delta, ms_a, n_a, ms_b, n_b, name in sorted(diffs, key=lambda x: -abs(x[0]))[:top]:
            print(f"  {delta:+8.1f} ms   A {ms_a:8.1f} ms {n_a:7d}x   B {ms_b:8.1f} ms {n_b:7d}x   {name[:80]}")


if __name__ == "__main__":
    main()
