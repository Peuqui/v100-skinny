"""Idle gaps between compute segments on one device of an nsys export (for TP runs without a pipeline: round turn = idle time
between two decode steps). Compute kernels are merged when closer than MERGE_US; a "step" is a merged segment longer than STEP_MS.
Usage: idle_gaps.py <sqlite> [device] [merge_us] [step_ms]"""
import sqlite3
import statistics
import sys

path = sys.argv[1]
device = int(sys.argv[2]) if len(sys.argv) > 2 else 0
merge_ns = float(sys.argv[3]) * 1000 if len(sys.argv) > 3 else 60_000
step_ns = float(sys.argv[4]) * 1e6 if len(sys.argv) > 4 else 5e6
INCLUDE_NCCL = len(sys.argv) > 5 and sys.argv[5] == "1"
db = sqlite3.connect(path)
rows = db.execute(
    """SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName
       WHERE k.deviceId = ? ORDER BY k.start""",
    (device,),
).fetchall()
segs: list[list[int]] = []
for s, e, n in rows:
    if "nccl" in n.lower() and not INCLUDE_NCCL:
        continue
    if segs and s - segs[-1][1] <= merge_ns:
        segs[-1][1] = max(segs[-1][1], e)
    else:
        segs.append([s, e])
steps = [x for x in segs if x[1] - x[0] > step_ns]
print(f"{path.rsplit('/', 1)[-1][-30:]} Gerät {device}: {len(rows)} Kernel, {len(segs)} Segmente, {len(steps)} Schritte > {step_ns / 1e6:.1f} ms")
if len(steps) >= 3:
    lens = [(e - s) / 1e6 for s, e in steps]
    gaps = [(b[0] - a[1]) / 1e6 for a, b in zip(steps, steps[1:])]
    print(
        f"  Schrittlänge ms: Median {statistics.median(lens):.2f} (min {min(lens):.2f}, max {max(lens):.2f}); "
        f"Lücke zum nächsten Schritt ms: Median {statistics.median(gaps):.3f} (min {min(gaps):.3f}, max {max(gaps):.3f})"
    )
