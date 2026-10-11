"""How much of a pipeline-parallel prefill runs on several GPUs at once.

Reads the CUPTI kernel table of an nsys SQLite export, merges each device's
kernel intervals into busy intervals, and reports per device the busy time and
for the whole capture how long 0, 1, 2, ... devices were busy at the same time.
A serial pipeline shows almost all busy time at concurrency 1; an overlapped
one shows a large share at 2 or more.

Usage: python nsys_stage_overlap.py <export.sqlite> [gap_ms]
gap_ms joins kernel intervals of one device closer than this (default 0.05 ms),
so launch gaps inside a stage's forward do not count as idle.
"""

import sqlite3
import sys
from collections import defaultdict


def busy_intervals(rows: list[tuple[int, int]], gap_ns: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(rows):
        if merged and start - merged[-1][1] <= gap_ns:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def main() -> None:
    path = sys.argv[1]
    gap_ns = int(float(sys.argv[2]) * 1e6) if len(sys.argv) > 2 else 50_000
    con = sqlite3.connect(path)
    per_device: dict[int, list[tuple[int, int]]] = defaultdict(list)
    # NCCL send/recv kernels spin while they wait for the peer stage, so they
    # would count waiting as busy; only compute kernels mark a working stage.
    for device, start, end in con.execute(
        "SELECT k.deviceId, k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k"
        " JOIN StringIds s ON s.id = k.shortName WHERE s.value NOT LIKE 'nccl%'"
    ):
        per_device[device].append((start, end))

    events: list[tuple[int, int]] = []
    first = min(start for rows in per_device.values() for start, _ in rows)
    last = max(end for rows in per_device.values() for _, end in rows)
    print(f"capture span {(last - first) / 1e9:.3f} s, gap join {gap_ns / 1e6} ms")
    for device in sorted(per_device):
        intervals = busy_intervals(per_device[device], gap_ns)
        busy = sum(end - start for start, end in intervals)
        kernel = sum(end - start for start, end in per_device[device])
        print(
            f"GPU {device}: kernel time {kernel / 1e9:.3f} s, busy {busy / 1e9:.3f} s,"
            f" {len(intervals)} busy intervals"
        )
        for start, end in intervals:
            events.append((start, 1))
            events.append((end, -1))

    events.sort()
    time_at: dict[int, int] = defaultdict(int)
    active = 0
    previous = first
    for timestamp, delta in events:
        time_at[active] += timestamp - previous
        active += delta
        previous = timestamp
    total = last - first
    for level in sorted(time_at):
        share = time_at[level] / total * 100
        print(f"{level} devices busy: {time_at[level] / 1e9:.3f} s ({share:.1f} %)")


if __name__ == "__main__":
    main()
