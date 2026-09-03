"""PP4-Schrittfixkosten aus dem Boot-61-nsys-SQLite attribuieren.

Idee: PP4 (device 4) = 8 Ziel-Layer + Drafter + Sampling + hc_head/lm_head.
Eine V100-Stufe (device 1) = 8 Ziel-Layer pur. Differenz der Kernel-Zeiten
je Schritt = Kosten des Drafter/Sampling-Anteils, kernel-genau.
"""
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
WINDOW_S = 60

t_max = db.execute("SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
t_min = t_max - int(WINDOW_S * 1e9)

rows = db.execute(f"""
    SELECT deviceId, s.value, SUM(k.end - k.start) / 1e6, COUNT(*)
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON s.id = k.shortName
    WHERE k.start >= {t_min}
    GROUP BY deviceId, s.value
""").fetchall()

per_dev: dict = {}
for dev, name, ms, n in rows:
    per_dev.setdefault(dev, {})[name.split("(")[0]] = (ms, n)

# Schritte im Fenster: Sparse-Decode-Launches der V100-Stufe / 8 Layer
v100 = per_dev[1]
pp4 = per_dev[4]
steps = v100["_sparse_attn_decode_ragged_kernel"][1] / 8
print(f"Schritte im 60-s-Fenster: {steps:.0f}")

names = sorted(set(v100) | set(pp4),
               key=lambda k: -(pp4.get(k, (0, 0))[0] - v100.get(k, (0, 0))[0]))
print(f"\n{'Kernel':44s} {'PP4 ms/St':>9s} {'V100 ms/St':>10s} "
      f"{'Extra ms/St':>11s} {'Extra L/St':>10s}")
tot_pp4 = tot_v100 = tot_extra = 0.0
for k in names:
    if "ncclDevKernel" in k:
        continue
    p_ms, p_n = pp4.get(k, (0, 0))
    v_ms, v_n = v100.get(k, (0, 0))
    d_ms = (p_ms - v_ms) / steps
    d_n = (p_n - v_n) / steps
    tot_pp4 += p_ms / steps
    tot_v100 += v_ms / steps
    tot_extra += d_ms
    if abs(d_ms) >= 0.05 or p_ms / steps >= 0.5:
        print(f"{k[:44]:44s} {p_ms / steps:9.2f} {v_ms / steps:10.2f} "
              f"{d_ms:11.2f} {d_n:10.1f}")
print(f"\n{'SUMME (ohne NCCL)':44s} {tot_pp4:9.2f} {tot_v100:10.2f} "
      f"{tot_extra:11.2f}")
nccl_pp4 = pp4.get("ncclDevKernel_SendRecv", (0, 0))[0] / steps
nccl_v100 = v100.get("ncclDevKernel_SendRecv", (0, 0))[0] / steps
print(f"{'NCCL SendRecv (Warten)':44s} {nccl_pp4:9.2f} {nccl_v100:10.2f}")
