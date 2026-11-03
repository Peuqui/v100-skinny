"""The longest individual kernel calls of one round segment (middle >3 ms merged compute segment) of one device, with their grid sizes.
Usage: longest_calls.py <sqlite> <device> [name_substring] [count=12]"""
import sqlite3
import sys

path, device = sys.argv[1], int(sys.argv[2])
needle = sys.argv[3] if len(sys.argv) > 3 else ""
count = int(sys.argv[4]) if len(sys.argv) > 4 else 12
db = sqlite3.connect(path)
rows = db.execute(
    """SELECT k.start, k.end, s.value, k.gridX, k.gridY, k.gridZ, k.blockX FROM CUPTI_ACTIVITY_KIND_KERNEL k
       JOIN StringIds s ON s.id = k.shortName WHERE k.deviceId = ? ORDER BY k.start""",
    (device,),
).fetchall()
segs: list[list[int]] = []
for s, e, n, *_ in rows:
    if "nccl" in n.lower():
        continue
    if segs and s - segs[-1][1] <= 60_000:
        segs[-1][1] = max(segs[-1][1], e)
    else:
        segs.append([s, e])
big = [x for x in segs if x[1] - x[0] > 3_000_000]
lo, hi = big[len(big) // 2]
inside = [r for r in rows if lo <= r[0] and r[1] <= hi and needle in r[2]]
inside.sort(key=lambda r: -(r[1] - r[0]))
for s, e, n, gx, gy, gz, bx in inside[:count]:
    print(f"{(s - lo) / 1e6:8.3f} ms  dauer {(e - s) / 1e3:8.1f} us  grid {gx}x{gy}x{gz} block {bx}  {n[:60]}")
