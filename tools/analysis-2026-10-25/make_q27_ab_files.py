"""Write the 27B metadata A/B files into ~/.cache/bench-scripts: the test entry (production entry with the test tree and its own cache copies)
and the A/B script (derived from ab_gdnmeta_gsm_2026-10-25.sh). Run once."""
import re
from pathlib import Path

import yaml

bench = Path("/home/mp/.cache/bench-scripts")
config = yaml.safe_load(open("/home/mp/.config/llama-swap/config.yaml"))
production = "Qwen3.8-27B-NVFP4-vllm"
test = "Qwen3.8-27B-NVFP4-meta-vllm"
text = open("/home/mp/.config/llama-swap/config.yaml").read()
block = re.search(rf"^  {re.escape(production)}:\n(?:    .*\n|\n)+", text, re.M).group(0)
block = block.replace(f"  {production}:\n", f"  {test}:\n", 1)
assert block.count(f"--served-model-name {production} ") == 1
block = block.replace(f"--served-model-name {production} ", f"--served-model-name {test} ", 1)
for old, new in (
    ("torchinductor-prod-2026-10-24", "torchinductor-q27meta-2026-10-25"),
    ("triton-cache-prod-2026-10-24", "triton-cache-q27meta-2026-10-25"),
    ("vllm-prod-2026-10-24", "vllm-q27meta-2026-10-25"),
):
    assert block.count(old) == 1, old
    block = block.replace(old, new)
old_path = "    - PYTHONPATH=/home/mp/.cache/triton-sm7x-prod-2026-10-23\n"
assert block.count(old_path) == 1
block = block.replace(old_path, old_path.rstrip("\n") + ":/home/mp/Projekte/vllm-research/1Cat-vLLM-q27-meta\n")
parsed = yaml.safe_load(block)
assert test in parsed and "1Cat-vLLM-q27-meta" in "".join(parsed[test]["env"])
(bench / "q27_meta_entry_2026-10-25.yaml").write_text(block)

script = (bench / "ab_gdnmeta_gsm_2026-10-25.sh").read_text()
pairs = [
    ("ab_glm_gdnmeta_", "ab_q27meta_"),
    ("vor-ab-gdnmeta-", "vor-ab-q27meta-"),
    ("glm53_gdnmeta_entry_2026-10-25.yaml", "q27_meta_entry_2026-10-25.yaml"),
    ("NAME_P=GLM-5.3-Flash-NVFP4-DFlash2-vllm", "NAME_P=Qwen3.8-27B-NVFP4-vllm"),
    ("NAME_T=GLM-5.3-Flash-NVFP4-DFlash2-gdnmeta-vllm", "NAME_T=Qwen3.8-27B-NVFP4-meta-vllm"),
    ("1Cat-vLLM-glm53-dev2", "1Cat-vLLM-q27-meta"),
    ("1Cat-vLLM-glm53-prod-s5", "1Cat-vLLM-work"),
    ("A/B gdnmeta", "A/B q27meta"),
    ("GLM gdnmeta A/B", "27B meta A/B"),
    ("gsm8k_api_client.py \"$OUT/$n-$arm-gsm8k.json\" 60", "gsm8k_api_client.py \"$OUT/$n-$arm-gsm8k.json\" 30"),
    ("|Fused speculative GDN metadata\"", "|Fused speculative GDN metadata|MTP shared GDN batch metadata\""),
]
for old, new in pairs:
    assert old in script, old
    script = script.replace(old, new)
# quality only on the first occurrence of every arm; the compare block is replaced by a T-against-P comparison after the loop
script = script.replace('  [ "$arm" = P ] && extras=0\n', "")
start = script.index("    python3 - \"$OUT/$n-$arm-quality.json\" <<'PY'")
end = script.index("PY\n", script.index("<<'PY'", start) + 10) + 3
script = script[:start] + script[end:]
tail_marker = "unload\necho \"##### fertig"
assert tail_marker in script
compare = """python3 - "$OUT" <<'PY'
import glob, json, sys
out = sys.argv[1]
def first(arm):
    files = sorted(glob.glob(f"{out}/*-{arm}-quality.json"))
    return {r["id"]: r for r in json.load(open(files[0]))} if files else None
t, p = first("T"), first("P")
if t and p:
    for k in p:
        print(f"[Textvergleich T gegen P] {k:9s} gleich: {t[k]['text'] == p[k]['text']}  Token {t[k]['tokens']} / {p[k]['tokens']}")
PY
"""
script = script.replace(tail_marker, compare + tail_marker)
header_old = script.split("\nset -u\n", 1)[0]
header_new = """#!/bin/bash
# 25.10.2026: A/B(/A/B) of the Qwen3.8-27B (MTP k=3, TP2 on the two RTX) production entry: P = installed entry (tree 1Cat-vLLM-work,
# prod-fork-main 1addd22db), T = the same entry with tree 1Cat-vLLM-q27-meta (prod-fork-main + the two gate commits of
# gdn-metadata-turing: shared and fused MTP GDN metadata also on Turing) and copies of the production compile caches.
# GSM8K-30 runs on EVERY arm (same questions, byte-equal tokens expected); the quality questions on the first T and the first P,
# compared with each other after the loop. Usage (from a copy): ab_q27meta_2026-10-25.sh <tag> <arm>...    arms: T P"""
script = script.replace(header_old, header_new, 1)
(bench / "ab_q27meta_2026-10-25.sh").write_text(script)
print("geschrieben: q27_meta_entry_2026-10-25.yaml, ab_q27meta_2026-10-25.sh")
