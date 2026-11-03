"""Add the 27B note to MEMORY.md and update the round-gap line."""
p = "/home/mp/.claude/projects/-home-mp-Projekte-vllm-research/memory/MEMORY.md"
s = open(p).read()
old = "- **[Rundenwechsel-Lücke Stufe 4 → Stufe 0: GLM 3,4 ms (behoben, +2,1…2,3 % Decode), DSv4 bei 150k Kontext 4,4 ms/78 ms offen, 27B-MTP gleicher Volta-Gate](project_round_turn_gap_2026-10-25.md)**"
new = "- **[Rundenwechsel-Lücke Stufe 4 → Stufe 0: GLM 3,4 ms (behoben, +2,1…2,3 % Decode), 27B-MTP (behoben, +5 %), DSv4 bei 150k Kontext 4,4 ms/78 ms offen](project_round_turn_gap_2026-10-25.md)**"
assert old in s
s = s.replace(old, new, 1)
marker = "## Zuerst lesen\n"
line = "- **[Qwen3.8-27B PRODUKTION seit 25.10. 15:29: MTP-GDN-Metadatenpfad auch auf Turing — Decode +5,1…5,6 %, GSM8K +4,9 %, Texte gleich, eigener Baum q27-meta, Tag verified-2026-10-25-q27-meta](project_q27_mtp_metadata_turing_2026-10-25.md)** — Rücksprung `config.yaml.vor-q27-prod-meta-2026-10-25`; Template lehnt reasoning_effort max ab\n"
assert line not in s
s = s.replace(marker, marker + line, 1)
open(p, "w").write(s)
print("ok")
