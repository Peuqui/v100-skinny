"""Add the scheduler note to MEMORY.md."""
p = "/home/mp/.claude/projects/-home-mp-Projekte-vllm-research/memory/MEMORY.md"
s = open(p).read()
marker = "## Zuerst lesen\n"
line = ("- **[Scheduler-Fix: unter PP kopierte/serialisierte der Scheduler je Decode-Schritt die ganze Token-Liste (nur V1 liest sie) — PRODUKTION seit 25.10. DSv4 16:49 + GLM 17:00: Schritt bei 287k −14,9 %, GLM 516k −15,5 %, kurz unverändert]"
        "(project_sched_token_history_2026-10-25.md)** — Zweig sched-v2-no-token-history; Trace-Methode `VLLM_SM70_ASYNC_CPU_TRACE=1`; Rücksprünge `config.yaml.vor-{dsv4,glm}-prod-sched-2026-10-25`\n")
assert line not in s
s = s.replace(marker, marker + line, 1)
open(p, "w").write(s)
print("ok")
