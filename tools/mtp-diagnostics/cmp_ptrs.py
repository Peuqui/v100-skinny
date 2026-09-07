#!/usr/bin/env python3
"""Vergleicht die Puffer-Adressen zwischen Graph-Aufzeichnung und Replay.

Der FULL-Graph-Replay beruht auf einem Vertrag: die Laufzeit muss in dieselben
Puffer schreiben, die beim Capture eingebacken wurden. Jedes Feld, dessen
Adresse sich zwischen cg.captured und cg.replay unterscheidet, ist fuer den
Graphen unsichtbar - er rechnet dort mit den Werten der Aufzeichnung weiter.
"""

import re
import sys
from pathlib import Path

FIELD = re.compile(r"(\S+?)=(@0x[0-9a-f]+/\w+\([^)]*\)/\S+|[^\s=]+)")


def parse(path: Path, tag: str) -> list[dict[str, str]]:
    rows = []
    for line in path.read_text().splitlines():
        if f" {tag} " not in line:
            continue
        body = line.split(f" {tag} ", 1)[1]
        rows.append(dict(FIELD.findall(body)))
    return rows


def key(row: dict[str, str]) -> tuple:
    return (row.get("mode"), row.get("toks"), row.get("reqs"), row.get("uni"))


def main() -> int:
    log = Path(sys.argv[1])
    captured = parse(log, "cg.captured")
    replays = parse(log, "cg.replay")
    print(f"aufgezeichnet: {len(captured)} Descs, Replays: {len(replays)}\n")

    by_key = {key(r): r for r in captured}
    print("=== aufgezeichnete Descs ===")
    for k in by_key:
        print(f"  mode={k[0]} toks={k[1]} reqs={k[2]} uni={k[3]}")

    seen = set()
    for rep in replays:
        k = key(rep)
        if k in seen:
            continue
        seen.add(k)
        print(f"\n=== Replay mode={k[0]} toks={k[1]} reqs={k[2]} uni={k[3]} ===")
        cap = by_key.get(k)
        if cap is None:
            print("  KEIN passender Capture-Desc gefunden!")
            print(f"  vorhanden: {sorted(by_key)}")
            continue
        fields = sorted(set(cap) | set(rep))
        same = diff = 0
        for f in fields:
            a, b = cap.get(f), rep.get(f)
            if a == b:
                same += 1
                continue
            diff += 1
            print(f"  ABWEICHUNG {f}:")
            print(f"    capture: {a}")
            print(f"    replay : {b}")
        print(f"  -> {same} identisch, {diff} abweichend")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
