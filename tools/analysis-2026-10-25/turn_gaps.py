"""Gap between the end of the last stage's compute and the start of stage 0's next round, for every round of an nsys export.
A round starts with a >3 ms compute segment on device 0 (kernels merged when <60 us apart); the last stage is the highest device id.
Usage: turn_gaps.py <sqlite>..."""
import sqlite3
import statistics
import sys

MERGE_NS = 60_000
BIG_NS = 3_000_000


def segments(intervals):
    out: list[list[int]] = []
    for s, e in intervals:
        if out and s - out[-1][1] <= MERGE_NS:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


for path in sys.argv[1:]:
    db = sqlite3.connect(path)
    rows = db.execute(
        """SELECT k.deviceId, k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN StringIds s ON s.id = k.shortName ORDER BY k.start"""
    ).fetchall()
    devices = sorted({r[0] for r in rows})
    first, last = devices[0], devices[-1]
    comp = {d: segments([(s, e) for dd, s, e, n in rows if dd == d and "nccl" not in n.lower()]) for d in devices}
    starts = [s for s, e in comp[first] if e - s > BIG_NS]
    last_ends = [e for s, e in comp[last] if e - s > BIG_NS]
    gaps = []
    for a, b in zip(starts, starts[1:]):
        ends = [e for e in last_ends if a < e <= b]
        if ends:
            gaps.append((b - max(ends)) / 1e6)
    periods = [(b - a) / 1e6 for a, b in zip(starts, starts[1:])]
    name = path.rsplit("/", 1)[-1][-42:]
    if gaps:
        print(
            f"{name}: Runden {len(gaps)}, Lücke Stufe {last} Ende -> Stufe {first} Anfang ms: "
            f"Median {statistics.median(gaps):6.2f}, min {min(gaps):6.2f}, max {max(gaps):6.2f}; "
            f"Rundenabstand Median {statistics.median(periods):7.2f} ms"
        )
    else:
        print(f"{name}: keine Runden erkannt")
