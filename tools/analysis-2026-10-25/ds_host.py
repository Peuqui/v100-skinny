"""Host-side view of the round turn in an nsys export: long CUDA runtime calls and NVTX execute ranges of every host thread
around the window [lo, hi] ms (same t0 convention as hops.py / turn_window.py: start of the mid >3 ms compute segment on device 0).
Usage: ds_host.py <sqlite> <lo_ms> <hi_ms>"""
import sqlite3
import sys

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
print("t0", t0)
a, b = t0 + int(lo * 1e6), t0 + int(hi * 1e6)
print("--- NVTX execute-Bereiche (relativ ms)")
for s, e, text, tid in db.execute(
    "select start, end, text, globalTid from NVTX_EVENTS where text like 'execute_context%' and end >= ? and start <= ? order by start", (a, b)
):
    print(f"  {(s - t0) / 1e6:9.3f} -> {(e - t0) / 1e6:9.3f} ms ({(e - s) / 1e6:6.3f})  tid {tid % 100000}  {text}")
print("--- lange Runtime-Aufrufe (> 40 us) je Thread")
for s, e, name, tid in db.execute(
    """select r.start, r.end, n.value, r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds n on n.id = r.nameId
       where r.start between ? and ? and r.end - r.start > 40000 order by r.start""", (a, b)
):
    print(f"  {(s - t0) / 1e6:9.3f} -> {(e - t0) / 1e6:9.3f} ms ({(e - s) / 1e3:8.1f} us)  tid {tid % 100000}  {name}")
print("--- Anzahl Launches je Thread im Fenster")
for tid, n in db.execute(
    """select globalTid, count(*) from CUPTI_ACTIVITY_KIND_RUNTIME where start between ? and ? group by globalTid order by 2 desc""", (a, b)
):
    print(f"  tid {tid % 100000}: {n}")
