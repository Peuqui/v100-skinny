"""Compare two py-spy raw (folded stack) recordings of the same workload.

Usage: python pyspy_raw_diff.py <A.raw> <B.raw> [top]
Lists the functions whose share of all samples changed most, as self time (leaf frame) and as inclusive time
(anywhere on the stack). Shares in percent of each recording's samples, so recordings of equal length compare.
"""
import sys
from collections import Counter


def load(path: str) -> tuple[Counter, Counter, int]:
    self_time: Counter = Counter()
    inclusive: Counter = Counter()
    total = 0
    for line in open(path):
        stack, _, count = line.rstrip("\n").rpartition(" ")
        if not stack or not count.isdigit():
            continue
        n = int(count)
        frames = [f.split(" (")[0] + " (" + f.split(" (")[1].split(":")[0] + ")" if " (" in f else f
                  for f in stack.split(";")]
        total += n
        self_time[frames[-1]] += n
        for frame in set(frames):
            inclusive[frame] += n
    return self_time, inclusive, total


def show(title: str, a: Counter, total_a: int, b: Counter, total_b: int, top: int) -> None:
    print(f"\n=== {title} (Anteil der Proben in %, B - A)")
    rows = []
    for key in set(a) | set(b):
        share_a = 100 * a.get(key, 0) / total_a
        share_b = 100 * b.get(key, 0) / total_b
        rows.append((share_b - share_a, share_a, share_b, key))
    for delta, share_a, share_b, key in sorted(rows, key=lambda r: -abs(r[0]))[:top]:
        print(f"  {delta:+6.2f}  A {share_a:6.2f}  B {share_b:6.2f}  {key[:120]}")


def main() -> None:
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    self_a, incl_a, total_a = load(sys.argv[1])
    self_b, incl_b, total_b = load(sys.argv[2])
    print(f"Proben: A {total_a}, B {total_b}")
    show("Eigenzeit", self_a, total_a, self_b, total_b, top)
    show("Gesamtzeit", incl_a, total_a, incl_b, total_b, top)


if __name__ == "__main__":
    main()
