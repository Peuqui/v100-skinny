"""Kernels of S0 and S4 (and all NCCL) in the window around the round turn, run-length grouped by name.
Usage: turn_window.py <sqlite> <lo_ms> <hi_ms>   (times relative to the start of a >3 ms segment on device 0 in mid-capture)"""
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
lo, hi = float(sys.argv[2]), float(sys.argv[3])
rows = db.execute("""SELECT k.deviceId, k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
                     JOIN StringIds s ON s.id = k.shortName ORDER BY k.start""").fetchall()
d0 = [(s, e) for d, s, e, n in rows if d == 0 and "nccl" not in n.lower()]
segs = []
for s, e in d0:
    if segs and s - segs[-1][1] <= 60_000:
        segs[-1][1] = max(segs[-1][1], e)
    else:
        segs.append([s, e])
big = [x for x in segs if x[1] - x[0] > 3_000_000]
t0 = big[len(big) // 2][0]
print("t0 abs ns", t0)
sel = [(d, s, e, n) for d, s, e, n in rows if d in (0, 4) and t0 + lo * 1e6 <= s <= t0 + hi * 1e6]
for dev in (4, 0):
    print(f"=== Stufe {dev}")
    cur = None
    for d, s, e, n in sel:
        if d != dev:
            continue
        key = n[:80]
        if cur and cur[0] == key:
            cur[2] = e; cur[3] += 1; cur[4] += e - s
        else:
            if cur:
                print(f"  {(cur[1]-t0)/1e6:8.3f}-{(cur[2]-t0)/1e6:8.3f} ms  x{cur[3]:3d}  Summe {cur[4]/1e3:7.1f} us  {cur[0]}")
            cur = [key, s, e, 1, e - s]
    if cur:
        print(f"  {(cur[1]-t0)/1e6:8.3f}-{(cur[2]-t0)/1e6:8.3f} ms  x{cur[3]:3d}  Summe {cur[4]/1e3:7.1f} us  {cur[0]}")
