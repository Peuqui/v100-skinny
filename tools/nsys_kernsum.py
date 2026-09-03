"""Per-Prozess-Kernel-Summen aus einem nsys-SQLite-Export.

Aufruf: python nsys_kernsum.py report.sqlite [fenster_sekunden]
Wertet die letzten N Sekunden der Aufzeichnung aus (Default: alles).
"""
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
window_s = float(sys.argv[2]) if len(sys.argv) > 2 else None

tables = {r[0] for r in db.execute(
    "SELECT name FROM sqlite_master WHERE type='table'")}
assert 'CUPTI_ACTIVITY_KIND_KERNEL' in tables, sorted(tables)

cols = {r[1] for r in db.execute(
    "PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
name_col = 'shortName' if 'shortName' in cols else 'demangledName'
pid_expr = ("(globalPid >> 24) & 0xFFFFFF" if 'globalPid' in cols
            else "processId")

t_max = db.execute(
    "SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
t_min = t_max - int(window_s * 1e9) if window_s else 0

rows = db.execute(f"""
    SELECT {pid_expr} AS pid, deviceId, s.value AS kname,
           SUM(k.end - k.start) / 1e6 AS ms, COUNT(*) AS n
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON s.id = k.{name_col}
    WHERE k.start >= {t_min}
    GROUP BY pid, deviceId, kname
""").fetchall()

per_proc: dict = {}
for pid, dev, kname, ms, n in rows:
    per_proc.setdefault((pid, dev), []).append((ms, n, kname))

for (pid, dev), ks in sorted(per_proc.items(), key=lambda kv: -kv[0][1]):
    total = sum(m for m, _, _ in ks)
    print(f"\n=== Prozess {pid} (device {dev}): GPU-Kernelzeit "
          f"{total:.0f} ms im Fenster, {len(ks)} Kernel-Arten ===")
    for ms, n, kname in sorted(ks, reverse=True)[:14]:
        short = kname.split('(')[0][:70]
        print(f"  {ms:9.1f} ms {n:9d}x  {short}")
