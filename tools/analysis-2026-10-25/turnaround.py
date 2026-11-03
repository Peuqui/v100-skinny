import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
t0 = int(sys.argv[2]); lo = float(sys.argv[3]); hi = float(sys.argv[4])  # ms relative to t0
rows = db.execute("""SELECT k.deviceId, k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
                     JOIN StringIds s ON s.id = k.shortName WHERE k.start BETWEEN ? AND ? ORDER BY k.start""",
                  (t0 + int(lo * 1e6), t0 + int(hi * 1e6))).fetchall()
for d, s, e, n in rows:
    if d in (0, 4) or "nccl" in n.lower():
        print(f"S{d} {(s - t0) / 1e6:8.3f} -> {(e - t0) / 1e6:8.3f} ms  ({(e - s) / 1e3:8.1f} us)  {n[:90]}")
