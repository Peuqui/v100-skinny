"""Flash-Next checkpoint comparison (RadixArk vs nvidia), production chat path.

Every request streams, so TTFT and decode rate come from the client:
decode tok/s = (completion_tokens - 1) / (total - ttft). Greedy, seed 1,
thinking on. Quality questions run behind the 13k context (a cached prefix
after the first one); the two speed probes run without it, the long one
behind a unique nonce so the prefix cache cannot serve its prefill.

    flashnext_deep_driver.py <workdir> <port> <context file> <max_tokens>
"""

import hashlib
import json
import sys
import time
import urllib.request
import uuid

W, PORT, CTX_FILE, MAXTOK = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
CTX = open(CTX_FILE).read()
TASK = "\n\nAufgabe, unabhängig vom Hintergrundmaterial oben: "

QUALITY = [
    ("q1_quanten", True, "Erkläre die Quantenphysik in 30 Sätzen."),
    ("q2_regenbogen", True, "Erkläre den Regenbogeneffekt in 30 Sätzen."),
    ("q3_kuanda_a", True, "Erkläre den Kuanda-Effekt in 30 Sätzen."),
    ("q3_kuanda_b", True, "Erkläre den Kuanda-Effekt in 30 Sätzen."),
    ("rechnen", True,
     "Ein Händler kauft 120 Äpfel für 0,35 € pro Stück. 15 % davon verderben. "
     "Den Rest verkauft er für 0,55 € pro Stück. Wie hoch ist sein Gewinn in "
     "Euro? Rechne nachvollziehbar."),
    ("logik", True,
     "Anna ist größer als Ben. Ben ist größer als Clara. Dora ist kleiner als "
     "Clara, aber größer als Emil. Wer ist die zweitkleinste Person? Begründe kurz."),
    ("code", True,
     "Schreibe eine Python-Funktion, die prüft, ob ein String eine gültige "
     "IPv4-Adresse ist, ohne das Modul ipaddress zu benutzen. Führende Nullen "
     "sind ungültig. Gib drei Testfälle mit erwartetem Ergebnis an."),
    ("fakten", True,
     "Nenne Jahr, Datum und Landestelle der ersten bemannten Mondlandung, die "
     "drei Astronauten der Mission und wer im Kommandomodul blieb."),
    ("zusammenfassung", True,
     "Fasse das Hintergrundmaterial oben in genau fünf Sätzen auf Deutsch zusammen."),
]


def stream(messages: list[dict], max_tokens: int) -> dict:
    body = json.dumps({
        "model": "qwen3.8-flash-next", "max_tokens": max_tokens, "temperature": 0, "seed": 1,
        "stream": True, "stream_options": {"include_usage": True},
        "messages": messages, "chat_template_kwargs": {"enable_thinking": True},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    content, reasoning, usage, finish = [], [], {}, None
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece_c = delta.get("content") or ""
                piece_r = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if (piece_c or piece_r) and ttft is None:
                    ttft = time.perf_counter() - t0
                content.append(piece_c)
                reasoning.append(piece_r)
                finish = choice.get("finish_reason") or finish
    total = time.perf_counter() - t0
    n = usage.get("completion_tokens", 0)
    decode = (n - 1) / (total - ttft) if ttft is not None and n > 1 and total > ttft else 0.0
    text = "".join(content)
    return {
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": n,
        "ttft_s": round(ttft or 0.0, 2), "total_s": round(total, 2),
        "decode_tok_s": round(decode, 2), "finish": finish,
        "prefill_tok_s": round(usage.get("prompt_tokens", 0) / ttft, 1) if ttft else None,
        "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
        "_content": text, "_reasoning": "".join(reasoning),
    }


def spec() -> tuple[float, float, float] | None:
    try:
        m = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=30).read().decode()
    except Exception:
        return None
    a = d = n = 0.0
    for line in m.splitlines():
        if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            a = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            d = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_drafts_total"):
            n = float(line.rsplit(" ", 1)[1])
    return a, d, n


def run(tag: str, messages: list[dict]) -> None:
    before = spec()
    row = stream(messages, MAXTOK)
    after = spec()
    if before and after:
        da, dd, dr = (x - y for x, y in zip(after, before))
        if dr:
            row["acc_len"] = round(da / dr + 1, 3)
    with open(f"{W}/text_{tag}.txt", "w") as f:
        f.write(f"<think>{row.pop('_reasoning')}</think>\n{row['_content']}")
    row.pop("_content")
    results[tag] = row
    with open(f"{W}/result.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  {tag}: {row}", flush=True)


results: dict[str, dict] = {}
stream([{"role": "user", "content": "Guten Tag."}], 8)  # warm-up
for tag, with_ctx, question in QUALITY:
    user = CTX + TASK + question if with_ctx else question
    run(tag, [{"role": "user", "content": user}])
run("tempo_kurz", [{"role": "user", "content": "Erkläre die Quantenphysik in 30 Sätzen."}])
nonce = f"Sitzungskennung {uuid.uuid4()}.\n\n"
run("tempo_lang", [{"role": "user", "content": nonce + CTX * 4 + TASK
                    + "Fasse das Hintergrundmaterial in drei Sätzen zusammen."}])
