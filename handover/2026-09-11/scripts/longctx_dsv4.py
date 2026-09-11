"""Long-context DFlash2 probe against the production llama-swap entry.

Streams each answer; decode rate = completion tokens / (total - time to
first token), the definition validated against llama.cpp on 2026-09-10.
"""
import hashlib, json, sys, time, urllib.request
label = sys.argv[1]
URL = "http://127.0.0.1:11435/v1/chat/completions"
MODEL = "DeepSeek-V4-Flash-nvfp4-DSpark-vllm"
ctx = open("/home/mp/Projekte/vllm-research/v100-skinny/tools/mtp-diagnostics/vorkontext.txt").read()
QS = ["Erklaere die Quantenphysik in 30 Saetzen. Die Aufgabe ist unabhaengig vom Hintergrundmaterial oben.",
      "Erklaere den Regenbogeneffekt in 30 Saetzen. Die Aufgabe ist unabhaengig vom Hintergrundmaterial oben.",
      "Erklaere den Coanda-Effekt in 30 Saetzen. Die Aufgabe ist unabhaengig vom Hintergrundmaterial oben."]
for i, q in enumerate(QS, 1):
    body = {"model": MODEL, "messages": [{"role": "system", "content": ctx}, {"role": "user", "content": q}],
            "max_tokens": 600, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.monotonic(); first = None; text = []; usage = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
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
    T = time.monotonic() - t0; n = usage.get("completion_tokens", 0)
    out = "".join(text)
    print(f"[{label} q{i}] Prompt {usage.get('prompt_tokens')} Tok | TTFT {first:.1f}s | {n} Tok | Decode {n/(T-first):.1f} tok/s | SHA {hashlib.sha256(out.encode()).hexdigest()[:16]}", flush=True)
