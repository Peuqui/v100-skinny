# Upstream-Beiträge — Entwürfe (Stand 2026-08-28)

Vorbereitet zur Freigabe durch Peuqui. NICHTS hiervon ist gepostet.
Alle externen Texte auf Englisch. Reihenfolge = empfohlene Veröffentlichung.

| # | Ziel | Art | Inhalt | Voraussetzung |
|---|------|-----|--------|---------------|
| 1 | vllm-project/vllm | Issue (+PR-Angebot) | PP+async+spec: Output-Trim läuft nur auf letzter Stufe (elif→if) | gegen aktuellen main verifizieren |
| 2 | dnv2003/v100-skinny | PR | Branch pp-mtp-merge (PP×TP+MTP, sm75-Paket, Device-0-Fixes, PLE-Kaskade) | GitHub-Fork unter Peuquis Account, Branch pushen |
| 3 | 1CatAI/1Cat-vLLM | Issue 1 | Capability-Gates fragen Device 0 statt aller sichtbaren GPUs | — |
| 4 | 1CatAI/1Cat-vLLM | Issue 2 | E5-Metadaten-Cache crasht an CSA/QSA-Modellen (shape [] vs [1]) | — |
| 5 | 1CatAI/1Cat-vLLM | Issue 3 | MTP-Profiling-Report bei PP blind (is_global_first_rank-Gate) | — |
| 6 | HF RadixArk/Qwen3.8-Flash-Next-NVFP4 | Discussion | Unquantisierter MTP-Block = Spekulation wird Verlustgeschäft auf Pre-Hopper | — |

Messgrundlage: QWEN4EXP-PORT-HANDOVER.md (Abschnitte 28.08.) und
MERGE-PROJECT-HANDOVER.md. Hardware: 2x Quadro RTX 8000 (sm75) + 3x
Tesla V100 (sm70), TP=2/PP=2, 1Cat-vLLM 1.3.0 + v100-skinny-Patches.
