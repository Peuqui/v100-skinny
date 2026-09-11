"""Capacity probe: one request near the configured max-model-len.

Streams the answer, stores text and usage, and reports TTFT and decode rate
(completion tokens / (total - TTFT), the definition validated 2026-09-10).
Usage: longctx_dsv4_long.py <url> <model> <repeats> <outfile>
"""
import hashlib, json, sys, time, urllib.request
url, model, repeats, outfile = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
ctx = open("/home/mp/Projekte/vllm-research/v100-skinny/tools/mtp-diagnostics/vorkontext.txt").read()
question = ("Erklaere den Regenbogeneffekt in 30 Saetzen. "
            "Die Aufgabe ist unabhaengig vom Hintergrundmaterial oben.")
body = {"model": model,
        "messages": [{"role": "system", "content": "\n\n".join([ctx] * repeats)},
                     {"role": "user", "content": question}],
        "max_tokens": 600, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.monotonic(); first = None; text = []; usage = {}
with urllib.request.urlopen(req, timeout=7200) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        if d.get("usage"):
            usage = d["usage"]
        for ch in d.get("choices", []):
            piece = (ch.get("delta") or {}).get("content") or ""
            if piece and first is None:
                first = time.monotonic() - t0
            text.append(piece)
total = time.monotonic() - t0; n = usage.get("completion_tokens", 0); out = "".join(text)
json.dump({"usage": usage, "ttft": first, "total": total, "text": out}, open(outfile, "w"),
          ensure_ascii=False, indent=1)
print(f"Prompt {usage.get('prompt_tokens')} Tok | TTFT {first:.1f}s | {n} Tok | "
      f"Decode {n / (total - first):.1f} tok/s | SHA {hashlib.sha256(out.encode()).hexdigest()[:16]}")
print(out[:600].replace("\n", " "))
