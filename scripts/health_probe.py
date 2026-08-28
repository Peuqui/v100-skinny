#!/usr/bin/env python3
"""Quantitative Prefill-Gesundheit: mittlere Logprob ueber die Prompt-Tokens
eines stark vorhersagbaren Musters. Gesund liegt bei ca. -1..-4, ein kaputter
Forward bei ca. -12..-15 (nahezu Gleichverteilung ueber das Vokabular)."""
import json, sys, urllib.request

URL, LABEL = sys.argv[1], sys.argv[2]
TIMEOUT = int(sys.argv[3]) if len(sys.argv) > 3 else 60
PROMPT = ("The capital of France is Paris. The capital of Germany is Berlin. "
          "The capital of Italy is Rome. The capital of Spain is")
body = {"model": "qwen3.8-flash-next", "prompt": PROMPT, "max_tokens": 4,
        "temperature": 0, "logprobs": 3, "prompt_logprobs": 0}
req = urllib.request.Request(f"{URL}/v1/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
    d = json.load(r)
c = d["choices"][0]
lps = [info["logprob"]
       for e in (c.get("prompt_logprobs") or []) if e
       for info in e.values()]
tail = lps[-12:]
mean = sum(tail) / len(tail) if tail else float("nan")
print(f"[{LABEL}] prefill-health mean_logprob(last12)={mean:.2f} "
      f"generated={c['text']!r} first_tokens={c['logprobs']['tokens']}")
