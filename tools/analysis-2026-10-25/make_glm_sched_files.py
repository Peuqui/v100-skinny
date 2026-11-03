"""Derive the GLM production files for the scheduler tree (1Cat-vLLM-glm53-prod-sched) from the prod-meta ones: entry (same caches, the tree
is the only difference), install script, the two acceptance scripts, the chain and the starter."""
from pathlib import Path

import yaml

bench = Path("/home/mp/.cache/bench-scripts")


def derive(src: str, dst: str, pairs: list[tuple[str, str]]) -> None:
    text = (bench / src).read_text()
    for old, new in pairs:
        assert old in text, (src, old)
        text = text.replace(old, new)
    (bench / dst).write_text(text)


entry = (bench / "glm53_prod_entry_meta_2026-10-25.yaml").read_text()
assert entry.count("1Cat-vLLM-glm53-prod-meta") == 1
entry = entry.replace("1Cat-vLLM-glm53-prod-meta", "1Cat-vLLM-glm53-prod-sched")
parsed = yaml.safe_load(entry)
assert "1Cat-vLLM-glm53-prod-sched" in "".join(next(iter(parsed.values()))["env"])
(bench / "glm53_prod_entry_sched_2026-10-25.yaml").write_text(entry)

install = (bench / "install_glm_prod_meta_entry_2026-10-25.sh").read_text()
for old, new in (
    ("glm53_prod_entry_meta_2026-10-25.yaml", "glm53_prod_entry_sched_2026-10-25.yaml"),
    ("config.yaml.vor-glm-prod-meta-2026-10-25", "config.yaml.vor-glm-prod-sched-2026-10-25"),
    ('assert "glm53-prod-meta" in', 'assert "glm53-prod-sched" in'),
):
    assert old in install, old
    install = install.replace(old, new)
lines = install.split("\n")
lines[1] = "# 25.10.2026: replace the installed GLM-5.3-Flash DFlash2 production entry (tree 1Cat-vLLM-glm53-prod-meta, 2ae98d460) by the same entry with the"
lines[2] = "# tree 1Cat-vLLM-glm53-prod-sched (31d73aefa: the scheduler no longer sends the token history to the V2 runner every decode step). Same caches."
(bench / "install_glm_prod_sched_entry_2026-10-25.sh").write_text("\n".join(lines))

common = [
    ("validate_glm_prod_meta_2026-10-25", "validate_glm_prod_sched_2026-10-25"),
    ("1Cat-vLLM-glm53-prod-meta", "1Cat-vLLM-glm53-prod-sched"),
]
derive("validate_glm_prod_meta_2026-10-25.sh", "validate_glm_prod_sched_2026-10-25.sh", common + [
    ("glm53-prod-validate", "glm53-prod-sched-validate"),
    ("validate_glm_prod2_2026-10-25/quality.json", "validate_glm_prod_meta_2026-10-25/quality.json"),
    ("[Textvergleich mit validate_glm_prod2]", "[Textvergleich mit validate_glm_prod_meta]"),
])
derive("validate_glm_prod_meta_max_2026-10-25.sh", "validate_glm_prod_sched_max_2026-10-25.sh", [
    ("glm53-prod-meta-max", "glm53-prod-sched-max"),
    ("1Cat-vLLM-glm53-prod-meta", "1Cat-vLLM-glm53-prod-sched"),
])
derive("run_glm_prod_meta_validation_2026-10-25.sh", "run_glm_prod_sched_validation_2026-10-25.sh", [
    ("validate_glm_prod_meta_2026-10-25_run.sh", "validate_glm_prod_sched_2026-10-25_run.sh"),
    ("validate_glm_prod_meta_2026-10-25/quality.json", "validate_glm_prod_sched_2026-10-25/quality.json"),
    ("validate_glm_prod_meta_max_2026-10-25_run.sh", "validate_glm_prod_sched_max_2026-10-25_run.sh"),
])
(bench / "start_glm_prod_sched_2026-10-25.sh").write_text("""#!/bin/bash
# 25.10.2026: install the GLM production entry with the scheduler tree (backup automatic) and start its acceptance chain as a systemd user unit
# from script copies. Run it while no model is booting.
set -eu
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
cd /home/mp/.cache/bench-scripts
bash install_glm_prod_sched_entry_2026-10-25.sh
sleep 3
for f in validate_glm_prod_sched_2026-10-25 validate_glm_prod_sched_max_2026-10-25; do
  cp $f.sh ${f}_run.sh
  chmod +x ${f}_run.sh
done
chmod +x run_glm_prod_sched_validation_2026-10-25.sh
: > validate_glm_prod_sched_2026-10-25.log
systemd-run --user --unit=glm-sched-validate-2026-10-25 --collect --working-directory=/home/mp/.cache/bench-scripts \\
  --setenv=PATH=/usr/local/bin:/usr/bin:/bin \\
  bash -c 'exec ./run_glm_prod_sched_validation_2026-10-25.sh >> validate_glm_prod_sched_2026-10-25.log 2>&1'
sleep 4
systemctl --user is-active glm-sched-validate-2026-10-25
date +%H:%M:%S
""")
print("geschrieben")
