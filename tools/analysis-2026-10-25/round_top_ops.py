"""Kernel time by name inside one steady-state round segment (the middle >3 ms merged compute segment) of one device of an nsys export.
Usage: round_top_ops.py <sqlite> <device> [top=25]"""
import sqlite3
import sys
from collections import defaultdict

path, device = sys.argv[1], int(sys.argv[2])
top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
db = sqlite3.connect(path)
rows = db.execute(
    """SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName
       WHERE k.deviceId = ? ORDER BY k.start""",
    (device,),
).fetchall()
segs: list[list[int]] = []
for s, e, n in rows:
    if "nccl" in n.lower():
        continue
    if segs and s - segs[-1][1] <= 60_000:
        segs[-1][1] = max(segs[-1][1], e)
    else:
        segs.append([s, e])
big = [x for x in segs if x[1] - x[0] > 3_000_000]
lo, hi = big[len(big) // 2]
total = defaultdict(lambda: [0, 0])
for s, e, n in rows:
    if lo <= s and e <= hi:
        total[n[:70]][0] += e - s
        total[n[:70]][1] += 1
span = (hi - lo) / 1e6
busy = sum(v[0] for v in total.values()) / 1e6
print(f"Gerät {device}: Rundensegment {span:.2f} ms, Kernzeit {busy:.2f} ms ({100 * busy / span:.0f} %), {sum(v[1] for v in total.values())} Kernel")
for name, (ns, count) in sorted(total.items(), key=lambda kv: -kv[1][0])[:top]:
    print(f"  {ns / 1e6:7.3f} ms  {100 * ns / 1e6 / span:5.1f} %  x{count:4d}  {name}")
