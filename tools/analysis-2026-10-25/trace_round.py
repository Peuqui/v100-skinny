"""Print every trace line between two consecutive decode steps of the last (or a chosen) decode burst, with times relative to the first
line, to see the engine loop cadence (how many empty iterations lie between two real decode steps and what they cost).
Usage: trace_round.py <journal.txt> [burst_from_end=1] [rounds=2]"""
import re
import sys

STAMP = re.compile(r"(\d\d):(\d\d):(\d\d\.\d+) ")
FIELD = re.compile(r"(\w+)=([^\s]+)")
rows = []
for line in open(sys.argv[1], errors="ignore"):
    if "SM70 async cpu trace" not in line:
        continue
    m = STAMP.search(line)
    if not m:
        continue
    t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    f = dict(FIELD.findall(line.split("SM70 async cpu trace", 1)[1]))
    rows.append((t, f))
from_end = int(sys.argv[2]) if len(sys.argv) > 2 else 1
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 2

decode_idx = [i for i, (t, f) in enumerate(rows) if 1 <= int(f.get("scheduled_tokens", 0)) < 100]
# group decode rows into bursts (gap > 1 s) and take the requested one from the end
bursts: list[list[int]] = []
for i in decode_idx:
    if bursts and rows[i][0] - rows[bursts[-1][-1]][0] <= 1.0:
        bursts[-1].append(i)
    else:
        bursts.append([i])
burst = [b for b in bursts if len(b) >= 20][-from_end]
start = burst[len(burst) // 2]
end = burst[len(burst) // 2 + rounds]
t0 = rows[start][0]
for i in range(start, end + 1):
    t, f = rows[i]
    keys = ("mode", "scheduled_tokens", "queue_len", "schedule_ms", "execute_submit_ms", "sample_submit_ms", "wait_ms", "update_ms", "total_ms")
    print(f"{(t - t0) * 1000:9.2f} ms  step {f.get('step'):>6s}  " + "  ".join(f"{k}={f[k]}" for k in keys if k in f))
