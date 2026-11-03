"""Derive the 27B production entry (production name, test tree and cache copies of the A/B arm) and its install script from the GLM ones."""
from pathlib import Path

import yaml

bench = Path("/home/mp/.cache/bench-scripts")
entry = (bench / "q27_meta_entry_2026-10-25.yaml").read_text()
assert entry.count("Qwen3.8-27B-NVFP4-meta-vllm") == 2  # key and served name
entry = entry.replace("Qwen3.8-27B-NVFP4-meta-vllm", "Qwen3.8-27B-NVFP4-vllm")
parsed = yaml.safe_load(entry)
assert list(parsed) == ["Qwen3.8-27B-NVFP4-vllm"]
env = "".join(parsed["Qwen3.8-27B-NVFP4-vllm"]["env"])
assert "1Cat-vLLM-q27-meta" in env and "q27meta-2026-10-25" in env
(bench / "q27_prod_entry_meta_2026-10-25.yaml").write_text(entry)

install = (bench / "install_glm_prod_meta_entry_2026-10-25.sh").read_text()
pairs = [
    ("glm53_prod_entry_meta_2026-10-25.yaml", "q27_prod_entry_meta_2026-10-25.yaml"),
    ("config.yaml.vor-glm-prod-meta-2026-10-25", "config.yaml.vor-q27-prod-meta-2026-10-25"),
    ('assert "glm53-prod-meta" in', 'assert "1Cat-vLLM-q27-meta" in'),
]
for old, new in pairs:
    assert old in install, old
    install = install.replace(old, new)
lines = install.split("\n")
lines[1] = "# 25.10.2026: replace the installed Qwen3.8-27B production entry (code from the editable tree 1Cat-vLLM-work, prod-fork-main 1addd22db) by the one"
lines[2] = "# with the frozen tree 1Cat-vLLM-q27-meta (806b1b2ee: shared and fused MTP GDN metadata also on Turing) and the compile caches of the A/B arm. Same model name."
(bench / "install_q27_prod_meta_entry_2026-10-25.sh").write_text("\n".join(lines))
print("geschrieben")
