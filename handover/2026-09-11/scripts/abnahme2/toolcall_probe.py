"""Tool-Call-Sonde gegen einen laufenden OpenAI-kompatiblen Server (llama-swap).
Regel vom 11.09.: ein Profil ist erst abgenommen nach einem ECHTEN Tool-Call.
Aufruf: toolcall_probe.py <PORT> <served-model-name> [out_dir]
Zwei Runden: (1) Frage, die das Werkzeug verlangt -> erwartet tool_calls mit
geparsten Argumenten; (2) Werkzeugergebnis zurueckgeben -> erwartet Antwort in
content, die das Ergebnis nennt."""
import json
import os
import sys
import time
import urllib.request

PORT, MODEL = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "."
os.makedirs(OUT, exist_ok=True)
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Aktuelles Wetter fuer eine Stadt abrufen.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Name der Stadt"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]


def ask(messages, mt=800):
    body = json.dumps({"model": MODEL, "messages": messages, "tools": TOOLS,
                       "tool_choice": "auto", "max_tokens": mt,
                       "temperature": 0, "seed": 1}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        d = json.load(resp)
    return time.perf_counter() - t0, d


msgs = [{"role": "user", "content": "Wie ist das Wetter gerade in Hamburg? Nutze das Werkzeug."}]
t1, d1 = ask(msgs)
open(f"{OUT}/round1.json", "w").write(json.dumps(d1, ensure_ascii=False, indent=1))
m1 = d1["choices"][0]["message"]
calls = m1.get("tool_calls") or []
print(f"RUNDE1 finish={d1['choices'][0].get('finish_reason')} tool_calls={len(calls)} "
      f"content_chars={len(m1.get('content') or '')} s={t1:.1f}", flush=True)
ok1 = False
for c in calls:
    fn = c.get("function", {})
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except ValueError:
        args = None
    print(f"  call: name={fn.get('name')} args={args}", flush=True)
    ok1 = ok1 or (fn.get("name") == "get_weather" and isinstance(args, dict) and "city" in args)
print("RUNDE1", "BESTANDEN" if ok1 else "DURCHGEFALLEN", flush=True)
if not ok1:
    print("content:", (m1.get("content") or "")[:300])
    sys.exit(1)

msgs.append({"role": "assistant", "content": m1.get("content") or "", "tool_calls": calls})
msgs.append({"role": "tool", "tool_call_id": calls[0].get("id") or "call_1", "name": "get_weather",
             "content": json.dumps({"city": "Hamburg", "temperature": 17, "unit": "celsius", "condition": "bewoelkt"})})
t2, d2 = ask(msgs)
open(f"{OUT}/round2.json", "w").write(json.dumps(d2, ensure_ascii=False, indent=1))
m2 = d2["choices"][0]["message"]
content = m2.get("content") or ""
ok2 = "17" in content and not m2.get("tool_calls")
print(f"RUNDE2 finish={d2['choices'][0].get('finish_reason')} content_chars={len(content)} s={t2:.1f}")
print("  antwort:", content[:240].replace("\n", " "))
print("RUNDE2", "BESTANDEN" if ok2 else "DURCHGEFALLEN")
sys.exit(0 if ok2 else 1)
