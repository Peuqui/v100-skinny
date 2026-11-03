"""Kernels in the idle window between two consecutive long steps on one device (all kernels count as busy for the step detection).
Usage: gap_kernels.py <sqlite> <device> <merge_us> <step_ms> <which_gap>"""
import sqlite3
import sys

path, device = sys.argv[1], int(sys.argv[2])
merge_ns = float(sys.argv[3]) * 1000
step_ns = float(sys.argv[4]) * 1e6
which = int(sys.argv[5])
db = sqlite3.connect(path)
rows = db.execute(
    """SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName
       WHERE k.deviceId = ? ORDER BY k.start""",
    (device,),
).fetchall()
segs: list[list[int]] = []
for s, e, n in rows:
    if segs and s - segs[-1][1] <= merge_ns:
        segs[-1][1] = max(segs[-1][1], e)
    else:
        segs.append([s, e])
steps = [x for x in segs if x[1] - x[0] > step_ns]
a, b = steps[which], steps[which + 1]
print(f"Schritt {which} endet, Schritt {which + 1} beginnt {(b[0] - a[1]) / 1e6:.3f} ms später (Schrittlänge {(a[1] - a[0]) / 1e6:.2f} ms)")
t0 = a[1]
window = [(s, e, n) for s, e, n in rows if a[1] - 300_000 <= s <= b[0] + 200_000]
cur = None
for s, e, n in window:
    key = n[:90]
    if cur and cur[0] == key and s - cur[2] < 80_000:
        cur[2] = e
        cur[3] += 1
        cur[4] += e - s
    else:
        if cur:
            print(f"  {(cur[1] - t0) / 1e6:8.3f}-{(cur[2] - t0) / 1e6:8.3f} ms  x{cur[3]:3d}  {cur[4] / 1e3:7.1f} us  {cur[0]}")
        cur = [key, s, e, 1, e - s]
if cur:
    print(f"  {(cur[1] - t0) / 1e6:8.3f}-{(cur[2] - t0) / 1e6:8.3f} ms  x{cur[3]:3d}  {cur[4] / 1e3:7.1f} us  {cur[0]}")
