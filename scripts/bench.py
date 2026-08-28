#!/usr/bin/env python3
"""Stationaerer Decode-Durchsatz: fester Prompt, feste Tokenzahl, ignore_eos,
damit alle Varianten exakt gleich viel Arbeit leisten. Erster Lauf ist Aufwaermen."""
import json, sys, time, urllib.request

URL, LABEL = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
PROMPT = "Explain, step by step and in detail, how a pipelined CPU executes instructions."

def run(n):
    body = {"model": "qwen3.8-flash-next", "prompt": PROMPT, "max_tokens": n,
            "temperature": 0, "ignore_eos": True}
    req = urllib.request.Request(f"{URL}/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    return time.time() - t0, d["usage"]["completion_tokens"]

run(16)                                   # Aufwaermen, Ergebnis verworfen
times = []
for _ in range(3):
    dt, toks = run(N)
    times.append(toks / dt)
best, mean = max(times), sum(times) / len(times)
print(f"[{LABEL}] decode {N} tok  mittel={mean:.1f} tok/s  best={best:.1f} tok/s  "
      f"einzeln={[f'{t:.1f}' for t in times]}")
