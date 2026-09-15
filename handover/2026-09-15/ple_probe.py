#!/usr/bin/env python3
"""PLE-Kaskade: Text-Hash und Decode-Tempo gegen einen laufenden Server.

Aufruf: ple_probe.py URL MODEL LABEL [OUT.json]
Greedy, ignore_eos, 260 Token je Prompt (Hash-Vergleiche tragen nur bis ~260
Token, siehe Memory reference_llm_nondeterminism_batch_invariance). Prompt 1
laeuft zweimal, damit die Referenz ihre eigene Wiederholbarkeit belegt.
Aussagekraeftig nur ohne parallele Last auf dem Server.
"""
import hashlib
import json
import sys
import time
import urllib.request

URL, MODEL, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = sys.argv[4] if len(sys.argv) > 4 else None
TOKENS = 260
PROMPTS = [
    "Explain, step by step and in detail, how a pipelined CPU executes instructions.",
    "Erklaere ausfuehrlich, wie ein Regenbogen entsteht und warum er rund ist.",
    "Write a Python function that merges overlapping intervals, then explain it.",
]


def complete(prompt: str, max_tokens: int) -> tuple[str, int, float]:
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0, "ignore_eos": True}
    req = urllib.request.Request(f"{URL}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return d["choices"][0]["text"], d["usage"]["completion_tokens"], time.time() - t0


complete(PROMPTS[0], 16)  # Aufwaermen, verworfen
runs = []
for index, prompt in [(0, PROMPTS[0]), (1, PROMPTS[1]), (2, PROMPTS[2]), (0, PROMPTS[0])]:
    text, tokens, seconds = complete(prompt, TOKENS)
    sha = hashlib.sha256(text.encode()).hexdigest()[:16]
    runs.append({"prompt": index, "sha": sha, "tokens": tokens,
                 "seconds": round(seconds, 3), "tok_s": round(tokens / seconds, 2),
                 "head": text[:120], "text": text})
    print(f"[{LABEL}] p{index} sha={sha} {tokens} tok {tokens / seconds:.1f} tok/s")
repeat_ok = runs[0]["sha"] == runs[3]["sha"]
print(f"[{LABEL}] Wiederholung p0 gleich: {repeat_ok}")
if OUT:
    json.dump({"label": LABEL, "model": MODEL, "url": URL, "tokens": TOKENS,
               "repeat_ok": repeat_ok, "runs": runs,
               "time": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(OUT, "w"), indent=1, ensure_ascii=False)
