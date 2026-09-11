"""Frageteil der Chat-Sonde gegen einen LAUFENDEN Flash-Next-Server (Port 8027):
/v1/chat/completions, Chat-Template, enable_thinking=true, Denkblock und
Antwort getrennt. Aufruf: chat_ask.py <W> <PORT> <SCR> <MAXTOK>"""
import hashlib
import json
import os
import sys
import time
import urllib.request

W, PORT, SCR, MAXTOK = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
os.makedirs(W, exist_ok=True)
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
CTX = open(f"{SCR}/vorkontext.txt").read()
FRAGEN = [
    ("q1", "Erklaere die Quantenphysik in 30 Saetzen."),
    ("q2", "Erklaere den Regenbogeneffekt in 30 Saetzen."),
    ("q3", "Erklaere den Kuanda-Effekt in 30 Saetzen."),   # Verschreiber ABSICHTLICH, nicht aendern
]


def ask(user, mt):
    b = json.dumps({"model": "qwen3.8-flash-next", "max_tokens": mt, "temperature": 0, "seed": 1,
                    "messages": [{"role": "user", "content": user}],
                    "chat_template_kwargs": {"enable_thinking": True}}).encode()
    r = urllib.request.Request(URL, data=b, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(r, timeout=3600) as resp:
        d = json.load(resp)
    return time.perf_counter() - t0, d


def spec():
    try:
        m = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=30).read().decode()
    except Exception:
        return None
    a = dr = n = 0.0
    for l in m.splitlines():
        if l.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            a = float(l.rsplit(" ", 1)[1])
        elif l.startswith("vllm:spec_decode_num_draft_tokens_total"):
            dr = float(l.rsplit(" ", 1)[1])
        elif l.startswith("vllm:spec_decode_num_drafts_total"):
            n = float(l.rsplit(" ", 1)[1])
    return a, dr, n


ask("Guten Tag.", 8)
res, prev = {}, spec()
for tag, frage in FRAGEN:
    user = CTX + "\n\nAufgabe, unabhaengig vom Hintergrundmaterial oben: " + frage
    t, d = ask(user, MAXTOK)
    open(f"{W}/raw_{tag}.json", "w").write(json.dumps(d, ensure_ascii=False, indent=1))
    msg = d["choices"][0]["message"]
    u = d["usage"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    row = {"prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
           "s": round(t, 2), "tok_per_s": round(u["completion_tokens"] / t, 2),
           "finish": d["choices"][0].get("finish_reason"),
           "reasoning_chars": len(reasoning), "content_chars": len(content),
           "sha256": hashlib.sha256(content.encode()).hexdigest()[:16]}
    cur = spec()
    if prev and cur:
        da, dd, dr = (x - y for x, y in zip(cur, prev))
        if dd:
            row["acc_rate"] = round(da / dd, 4)
        if dr:
            row["acc_len"] = round(da / dr + 1, 3)
    prev = cur
    res[tag] = row
    open(f"{W}/text_{tag}.txt", "w").write(f"<think>{reasoning}</think>\n{content}")
    print(f"  {tag}: {row}", flush=True)
open(W + "/result.json", "w").write(json.dumps(res, indent=2))
