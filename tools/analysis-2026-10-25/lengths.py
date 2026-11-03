import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
for tag in ("ppv8-kc16-many-ds-fm-once", "ppv7-cap64-many-ds-fm-once"):
    match = re.search(rf"##### {tag} \(.*?(?=##### \S+ \(|##### fertig)", text, re.S)
    if not match:
        print(tag, "nicht gefunden")
        continue
    rows = re.findall(r"lang-(\d+)\s+prosa-lang:\s+(\d+) prompt tok, ttft\s+([\d.]+)s", match.group(0))
    cold = rows[0::2]
    print(f"== {tag}: {len(cold)} kalte Prefills")
    print(" Seed  Token  mod128 mod64  ttft s")
    for seed, tokens, ttft in cold:
        n = int(tokens)
        flag = "  <-- langsam" if float(ttft) > 7.15 else ""
        print(f" {seed}  {n}  {n % 128:5d} {n % 64:5d}  {float(ttft):5.1f}{flag}")
