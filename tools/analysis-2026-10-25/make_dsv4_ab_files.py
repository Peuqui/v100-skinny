"""Write the DeepSeek-V4 scheduler A/B files into ~/.cache/bench-scripts: the test entry (production entry, test tree first on PYTHONPATH,
the production caches unchanged: the change is not part of any compiled artifact) and the A/B script (derived from ab_q27meta_2026-10-25.sh,
GSM8K replaced by the long-context probe)."""
import re
from pathlib import Path

import yaml

bench = Path("/home/mp/.cache/bench-scripts")
production = "DeepSeek-V4-Flash-284B-A13B-MXFP4-FP8-DSpark-vllm"
test = "DeepSeek-V4-Flash-284B-A13B-MXFP4-FP8-DSpark-sched-vllm"
text = open("/home/mp/.config/llama-swap/config.yaml").read()
block = re.search(rf"^  {re.escape(production)}:\n(?:    .*\n|\n)+", text, re.M).group(0)
block = block.replace(f"  {production}:\n", f"  {test}:\n", 1)
assert block.count(f"--served-model-name {production} ") == 1
block = block.replace(f"--served-model-name {production} ", f"--served-model-name {test} ", 1)
old_path = "    - PYTHONPATH=/home/mp/.cache/triton-sm7x-prod-2026-10-23\n"
assert block.count(old_path) == 1
block = block.replace(old_path, old_path.rstrip("\n") + ":/home/mp/Projekte/vllm-research/1Cat-vLLM-dsv4-sched\n")
parsed = yaml.safe_load(block)
assert test in parsed and "1Cat-vLLM-dsv4-sched" in "".join(parsed[test]["env"])
(bench / "dsv4_sched_entry_2026-10-25.yaml").write_text(block)

script = (bench / "ab_q27meta_2026-10-25.sh").read_text()
pairs = [
    ("ab_q27meta_", "ab_dsv4sched_"),
    ("vor-ab-q27meta-", "vor-ab-dsv4sched-"),
    ("q27_meta_entry_2026-10-25.yaml", "dsv4_sched_entry_2026-10-25.yaml"),
    ("NAME_P=Qwen3.8-27B-NVFP4-vllm", f"NAME_P={production}"),
    ("NAME_T=Qwen3.8-27B-NVFP4-meta-vllm", f"NAME_T={test}"),
    ("1Cat-vLLM-q27-meta", "1Cat-vLLM-dsv4-sched"),
    ("A/B q27meta", "A/B dsv4sched"),
    ("27B meta A/B", "DSv4 sched A/B"),
    ("|Fused speculative GDN metadata|MTP shared GDN batch metadata\"", "\""),
]
for old, new in pairs:
    assert old in script, old
    script = script.replace(old, new)
gsm = [line for line in script.splitlines() if "gsm8k_api_client.py" in line]
assert len(gsm) == 1
longctx = ('  timeout 5400 python3 longctx_decode_probe.py $name dsv4sched-$arm$n > "$OUT/$n-$arm-longctx.txt" 2>&1; '
           'tail -n 5 "$OUT/$n-$arm-longctx.txt" | sed "s/^/[$arm$n longctx] /"')
script = script.replace(gsm[0], longctx)
header_old = script.split("\nset -u\n", 1)[0]
header_new = """#!/bin/bash
# 25.10.2026: A/B(/A/B) of the DeepSeek-V4 (PP5, DSpark) production entry: P = installed entry (tree 1Cat-vLLM-work, prod-fork-main 1addd22db),
# T = the same entry with tree 1Cat-vLLM-dsv4-sched (prod-fork-main + sched-v2-no-token-history: the scheduler no longer copies and serializes
# the complete token list of the request every decode step under PP with the V2 runner). Same compile caches for both arms. Per arm: prefill and
# decode client, long-context probe (30k / ~180k / ~290k tokens; ms per step), quality questions on the first T and the first P (compared
# after the loop). Usage (from a copy): ab_dsv4sched_2026-10-25.sh <tag> <arm>...    arms: T P"""
script = script.replace(header_old, header_new, 1)
(bench / "ab_dsv4sched_2026-10-25.sh").write_text(script)
print("geschrieben: dsv4_sched_entry_2026-10-25.yaml, ab_dsv4sched_2026-10-25.sh")
