"""Reproduktion des DeepSeek-Haengers bei der Tool-Ergebnis-Runde (12.09.).
Reihenfolge: (0) Boot per Kurzfrage, (1) Zwei-Runden-Chat OHNE tools,
(2) Tool-Call Runde 1, (3) Tool-Ergebnis Runde 2. Jede Runde mit eigenem
Timeout; ein Timeout/500 wird protokolliert, nicht abgebrochen.
Aufruf: toolcall_repro.py <PORT> <model> <out_dir>"""
import json, os, sys, time, urllib.request, urllib.error
PORT, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
TOOLS = [{"type": "function", "function": {"name": "get_weather",
          "description": "Aktuelles Wetter fuer eine Stadt abrufen.",
          "parameters": {"type": "object", "properties": {
              "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
              "required": ["city"]}}}]

def ask(tag, messages, tools=None, mt=200, timeout=420):
    body = {"model": MODEL, "messages": messages, "max_tokens": mt, "temperature": 0, "seed": 1}
    if tools: body["tools"] = tools; body["tool_choice"] = "auto"
    open(f"{OUT}/{tag}_request.json", "w").write(json.dumps(body, ensure_ascii=False, indent=1))
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            d = json.load(resp)
    except urllib.error.HTTPError as e:
        print(f"{tag}: HTTP {e.code} nach {time.perf_counter()-t0:.0f}s: {e.read()[:200]!r}", flush=True); return None
    except Exception as e:
        print(f"{tag}: FEHLER {type(e).__name__} nach {time.perf_counter()-t0:.0f}s: {e}", flush=True); return None
    open(f"{OUT}/{tag}_response.json", "w").write(json.dumps(d, ensure_ascii=False, indent=1))
    m = d["choices"][0]["message"]; u = d["usage"]
    print(f"{tag}: ok {time.perf_counter()-t0:.1f}s finish={d['choices'][0].get('finish_reason')} "
          f"prompt={u['prompt_tokens']} out={u['completion_tokens']} tool_calls={len(m.get('tool_calls') or [])} "
          f"content={(m.get('content') or '')[:90]!r}", flush=True)
    return d

ask("boot", [{"role": "user", "content": "Guten Tag."}], mt=8, timeout=1800)
# (1) zwei Runden ohne tools
m1 = [{"role": "user", "content": "Nenne drei Farben."}]
d = ask("plain1", m1)
if d:
    m1.append({"role": "assistant", "content": d["choices"][0]["message"].get("content") or ""})
    m1.append({"role": "user", "content": "Und welche davon ist die hellste?"})
    ask("plain2", m1)
# (2) Tool-Call
m2 = [{"role": "user", "content": "Wie ist das Wetter gerade in Hamburg? Nutze das Werkzeug."}]
d = ask("tool1", m2, tools=TOOLS)
if d and (d["choices"][0]["message"].get("tool_calls")):
    calls = d["choices"][0]["message"]["tool_calls"]
    m2.append({"role": "assistant", "content": d["choices"][0]["message"].get("content") or "", "tool_calls": calls})
    m2.append({"role": "tool", "tool_call_id": calls[0].get("id") or "call_1", "name": "get_weather",
               "content": json.dumps({"city": "Hamburg", "temperature": 17, "unit": "celsius", "condition": "bewoelkt"})})
    ask("tool2", m2, tools=TOOLS)
    # (4) Kontrolle: dieselbe Runde 2, aber OHNE tools-Feld in der Anfrage
    ask("tool2_ohne_tools", m2)
print("REPRO-ENDE", flush=True)
