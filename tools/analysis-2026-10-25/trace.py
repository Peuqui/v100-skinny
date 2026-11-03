import collections
import re
import sys

pattern = re.compile(r"^(\w+ \d+ [\d:]+) .*?\[TM_GEMM_TRACE\] desc=(\S+) policy=(\S+) measured=(\d)")
pattern2 = re.compile(r"(\w{3} +\d+ [\d:]+) .*desc=(sm70_\S+) policy=(\S+) measured=(\d)")
rows = []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    match = pattern2.search(line)
    if match:
        time, desc, policy, measured = match.groups()
        shape = desc.rsplit("_", 2)[-2:]
        m, n, k = shape[0].split("x")
        rows.append((time, int(m), int(n), int(k), int(shape[1]), policy, int(measured)))
print("Trace-Zeilen:", len(rows))
by_shape = collections.Counter((n, k, num) for _, _, n, k, num, _, _ in rows)
print("Formen (N, K, num):", dict(by_shape))
by_policy = collections.Counter((policy, measured) for *_, policy, measured in rows)
print("Policy/measured:", dict(by_policy))
print("\nZeitverlauf (Sekunde: Policy/measured → M-Werte), nur Zeilen mit measured=1 oder policy != default:")
last = None
for time, m, n, k, num, policy, measured in rows:
    if measured or policy != "default":
        print(f"  {time} M={m:5d} N={n} K={k} num={num} {policy} measured={measured}")
        last = time
