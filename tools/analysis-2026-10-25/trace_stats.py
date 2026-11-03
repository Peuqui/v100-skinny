"""Statistics of the engine's 'SM70 async cpu trace' lines (journal, short-precise stamps) per decode burst.
Lines: 'step=N mode=enqueue_return|block_result total_ms=.. schedule_ms=.. execute_submit_ms=.. sample_submit_ms=.. [wait_ms=.. update_ms=..]
queue_len=.. queue_size=.. scheduled_tokens=..'. A burst = consecutive lines with scheduled_tokens 1..99 (decode) closer than 1 s.
Usage: trace_stats.py <journal.txt>"""
import re
import statistics
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
    fields = dict(FIELD.findall(line.split("SM70 async cpu trace", 1)[1]))
    try:
        rows.append((t, fields["mode"], int(fields.get("scheduled_tokens", 0)),
                     {k: float(v) for k, v in fields.items() if k.endswith("_ms")},
                     int(fields.get("queue_len", 0)), fields.get("oldest_done")))
    except (KeyError, ValueError):
        continue
print(f"{len(rows)} Spurzeilen")

bursts: list[list[tuple]] = []
cur: list[tuple] = []
last = None
for row in rows:
    t, mode, tokens, ms, queue, done = row
    is_decode = 1 <= tokens < 100
    if is_decode:
        if cur and last is not None and t - last > 1.0:
            bursts.append(cur)
            cur = []
        cur.append(row)
        last = t
    elif tokens == 0:
        if cur and last is not None and t - last > 1.0:
            bursts.append(cur)
            cur = []
if cur:
    bursts.append(cur)


def med(values):
    return f"{statistics.median(values):7.3f}" if values else "      -"


for index, burst in enumerate(bursts):
    if len(burst) < 20:
        continue
    modes = [r[1] for r in burst]
    sched = [r[3].get("schedule_ms") for r in burst if "schedule_ms" in r[3]]
    total = [r[3].get("total_ms") for r in burst if "total_ms" in r[3]]
    submit = [r[3].get("execute_submit_ms") for r in burst if "execute_submit_ms" in r[3]]
    wait = [r[3]["wait_ms"] for r in burst if "wait_ms" in r[3]]
    update = [r[3]["update_ms"] for r in burst if "update_ms" in r[3]]
    blocks = [r for r in burst if r[1] == "block_result"]
    intervals = [b[0] - a[0] for a, b in zip(blocks, blocks[1:])]
    print(
        f"Burst {index}: {len(burst)} Zeilen ({modes.count('enqueue_return')} enqueue_return, {modes.count('block_result')} block_result), "
        f"Länge {burst[-1][0] - burst[0][0]:.2f} s | Median ms: schedule {med(sched)}, execute_submit {med(submit)}, wait {med(wait)}, "
        f"update {med(update)}, total {med(total)} | Abstand der block_result-Zeilen {med([i * 1000 for i in intervals])} ms | "
        f"queue_len Median {statistics.median(r[4] for r in burst):.0f}"
    )
