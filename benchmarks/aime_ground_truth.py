"""Ground truth for the three ninfer AIME 2026 fixtures.

Scoring a model against its own output is circular, so each answer here is
derived independently: f01 algebraically, f30 by exhaustive enumeration, f15
by exact-cover search. All three are cheap enough to compute exactly.
"""
from fractions import Fraction
from itertools import product


def f01():
    """Patrick/Tanya/Jose. d = m/n miles, find m+n.

    p*T = (p+2)(T-1) = (p+9)(T-2)  ->  p = 2T-2 and 2p = 9T-18.
    """
    # 2(2T-2) = 9T-18  ->  4T-4 = 9T-18  ->  T = 14/5
    T = Fraction(14, 5)
    p = 2 * T - 2
    d = p * T
    assert p * T == (p + 2) * (T - 1) == (p + 9) * (T - 2), "constraints"
    return d.numerator + d.denominator, f"d = {d} miles"


def f30():
    """Ordered 7-tuples in {1,2,3}^7 with sum = 0 mod 3 and the cyclic
    expression sum_k a_k a_{k+1} a_{k+3} = 0 mod 3 (indices mod 7)."""
    n = 0
    for t in product((1, 2, 3), repeat=7):
        if sum(t) % 3:
            continue
        s = sum(t[k] * t[(k + 1) % 7] * t[(k + 3) % 7] for k in range(7))
        if s % 3 == 0:
            n += 1
    return n, "exhaustive over 3^7 = 2187"


def _rings(N):
    """Every a x b cell loop in an N x N grid, as a frozenset of cells.

    The loop is the 2a+2b-4 boundary cells of an a x b rectangle; for a==2 or
    b==2 that is the whole rectangle.
    """
    out = []
    for a in range(2, N + 1):
        for b in range(2, N + 1):
            for r in range(N - a + 1):
                for c in range(N - b + 1):
                    cells = set()
                    for i in range(r, r + a):
                        for j in range(c, c + b):
                            if i in (r, r + a - 1) or j in (c, c + b - 1):
                                cells.add(i * N + j)
                    assert len(cells) == 2 * a + 2 * b - 4
                    out.append(frozenset(cells))
    return out


def f15(N=10, pieces=5):
    """Partitions of an N x N grid into exactly `pieces` cell loops."""
    rings = _rings(N)
    by_cell = [[] for _ in range(N * N)]
    for rg in rings:
        by_cell[min(rg)].append(rg)          # index by lowest-numbered cell

    total = N * N
    count = 0

    def rec(covered, used):
        nonlocal count
        if len(covered) == total:
            if used == pieces:
                count += 1
            return
        if used == pieces:
            return
        # lowest uncovered cell must be the minimum of some ring
        cell = next(i for i in range(total) if i not in covered)
        for rg in by_cell[cell]:
            if rg & covered:
                continue
            rec(covered | rg, used + 1)

    rec(frozenset(), 0)
    return count, f"exact cover of {N}x{N} by {pieces} loops"


if __name__ == "__main__":
    for name, fn in (("aime26_01", f01), ("aime26_30", f30),
                     ("aime26_15", f15)):
        ans, how = fn()
        print(f"{name}: {ans}    ({how})")
