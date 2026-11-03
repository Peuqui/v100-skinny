"""Busy segments per stage in the steady-state decode rounds of an nsys export: compute kernels (no NCCL) merged when
less than MERGE_US apart; prints the timeline of a few consecutive rounds relative to the first, to see the gaps at the hops."""
import sqlite3, sys, statistics
path = sys.argv[1]
MERGE_US = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
db = sqlite3.connect(path)
rows = db.execute("""SELECT k.deviceId, k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
                     JOIN StringIds s ON s.id = k.shortName ORDER BY k.start""").fetchall()
devs = sorted({r[0] for r in rows})
print("Geräte:", devs, "Kernel:", len(rows))
by = {d: [(s, e, n) for dd, s, e, n in rows if dd == d] for d in devs}
nccl = {d: [(s, e) for s, e, n in by[d] if "nccl" in n.lower()] for d in devs}
comp = {d: [(s, e) for s, e, n in by[d] if "nccl" not in n.lower()] for d in devs}
def segments(iv):
    out = []
    for s, e in iv:
        if out and s - out[-1][1] <= MERGE_US * 1000:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
seg = {d: segments(comp[d]) for d in devs}
for d in devs:
    big = [x for x in seg[d] if x[1] - x[0] > 300_000]
    print(f"Gerät {d}: {len(seg[d])} Segmente, davon {len(big)} > 0,3 ms; NCCL-Kernel {len(nccl[d])}")
# steady-state window: take segments > 3 ms on device 0 as round work; print 4 consecutive rounds from the middle
big0 = [x for x in seg[devs[0]] if x[1] - x[0] > 3_000_000]
print("Segmente > 3 ms auf Gerät", devs[0], ":", len(big0))
mid = len(big0) // 2
t0 = big0[mid][0]
print("Zeitleiste (ms ab Rundenanfang auf Gerät 0), Segmente > 0,3 ms, 3 Runden:")
t_end = big0[min(mid + 3, len(big0) - 1)][0]
for d in devs:
    line = []
    for s, e in seg[d]:
        if t0 - 2_000_000 <= s <= t_end and e - s > 300_000:
            line.append(f"{(s - t0) / 1e6:7.2f}-{(e - t0) / 1e6:7.2f}")
    print(f"S{d}: " + "  ".join(line))
