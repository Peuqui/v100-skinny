"""Derive the DeepSeek-V4 production entry (production name, test tree 1Cat-vLLM-dsv4-sched), its install script, acceptance script and starter."""
from pathlib import Path

import yaml

bench = Path("/home/mp/.cache/bench-scripts")
prod = "DeepSeek-V4-Flash-284B-A13B-MXFP4-FP8-DSpark-vllm"
test = "DeepSeek-V4-Flash-284B-A13B-MXFP4-FP8-DSpark-sched-vllm"

entry = (bench / "dsv4_sched_entry_2026-10-25.yaml").read_text()
assert entry.count(test) == 2
entry = entry.replace(test, prod)
parsed = yaml.safe_load(entry)
assert list(parsed) == [prod] and "1Cat-vLLM-dsv4-sched" in "".join(parsed[prod]["env"])
(bench / "dsv4_prod_entry_sched_2026-10-25.yaml").write_text(entry)

install = (bench / "install_q27_prod_meta_entry_2026-10-25.sh").read_text()
for old, new in (
    ("q27_prod_entry_meta_2026-10-25.yaml", "dsv4_prod_entry_sched_2026-10-25.yaml"),
    ("config.yaml.vor-q27-prod-meta-2026-10-25", "config.yaml.vor-dsv4-prod-sched-2026-10-25"),
    ('assert "1Cat-vLLM-q27-meta" in', 'assert "1Cat-vLLM-dsv4-sched" in'),
):
    assert old in install, old
    install = install.replace(old, new)
lines = install.split("\n")
lines[1] = "# 25.10.2026: replace the installed DeepSeek-V4 production entry (code from the editable tree 1Cat-vLLM-work, prod-fork-main 1addd22db) by the one"
lines[2] = "# with the frozen tree 1Cat-vLLM-dsv4-sched (f4faec141: the scheduler no longer sends the token history to the V2 runner every decode step). Same name and caches."
(bench / "install_dsv4_prod_sched_entry_2026-10-25.sh").write_text("\n".join(lines))

validate = (bench / "validate_q27_prod_meta_2026-10-25.sh").read_text()
for old, new in (
    ("Qwen3.8-27B-NVFP4-vllm", prod),
    ("validate_q27_prod_meta_2026-10-25", "validate_dsv4_prod_sched_2026-10-25"),
    ("1Cat-vLLM-q27-meta", "1Cat-vLLM-dsv4-sched"),
    ("Worker_TP", "Worker_PP"),
    ("vllm-research 27B Kontrolllauf", "vllm-research DSv4 Kontrolllauf"),
    ("ab_q27meta_a/1-P-quality.json", "ab_dsv4sched_a/1-P-quality.json"),
    ("q27-prod-meta-max", "dsv4-prod-sched-max"),
):
    assert old in validate, old
    validate = validate.replace(old, new)
validate = "\n".join(line for line in validate.split("\n") if "gsm8k_api_client.py" not in line)
validate = validate.replace(
    "# 25.10.2026: boot the installed Qwen3.8-27B production entry (tree 1Cat-vLLM-q27-meta: prod-fork-main 1addd22db + the Turing metadata gates)",
    "# 25.10.2026: boot the installed DeepSeek-V4 production entry (tree 1Cat-vLLM-dsv4-sched: prod-fork-main 1addd22db + the scheduler change)",
)
validate = validate.replace("GSM8K-30, ", "").replace("eight quality\n# questions (1500 tokens, compared byte for byte with the production arm of the A/B ab_dsv4sched_a)",
                                                       "eight quality\n# questions (1500 tokens, compared byte for byte with the production arm of the A/B ab_dsv4sched_a)")
(bench / "validate_dsv4_prod_sched_2026-10-25.sh").write_text(validate)

starter = (bench / "start_q27_prod_meta_2026-10-25.sh").read_text()
for old, new in (
    ("install_q27_prod_meta_entry_2026-10-25.sh", "install_dsv4_prod_sched_entry_2026-10-25.sh"),
    ("validate_q27_prod_meta_2026-10-25", "validate_dsv4_prod_sched_2026-10-25"),
    ("q27-meta-validate-2026-10-25", "dsv4-sched-validate-2026-10-25"),
):
    assert old in starter, old
    starter = starter.replace(old, new)
starter = starter.replace("27B production entry with the Turing metadata gates", "DeepSeek-V4 production entry with the scheduler change")
(bench / "start_dsv4_prod_sched_2026-10-25.sh").write_text(starter)
print("geschrieben")
