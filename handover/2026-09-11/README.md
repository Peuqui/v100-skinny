# Arbeitsdateien der Runde 10./11.09.2026

Gesichert aus dem sitzungsgebundenen Scratchpad. Die Skripte arbeiten
relativ zu ihrem eigenen Ordner (`S=$(dirname "$0")`), laufen also von hier
aus. Übergabe und Reihenfolge: `../../HANDOVER.md`.

## patches/ (gegen den Worktree `1Cat-vLLM-work`)

| Datei | Stand |
|---|---|
| `cleanup_neutral.diff` | Befunde 1, 3, 4, 5, 7, 8 — **committet** `43ccb9b8` (11.09. abends) |
| `cleanup_befund6.diff` | Befund 6 (`kv_cache_utils.py`) — **committet** `43ccb9b8` (11.09. abends) |
| `befund2.diff` | Befund 2 (`speculative.py`, index_share) — freigegeben, **nicht angewendet** |
| `punkt15_skinny_per_arch.diff` | Skinny-Build pro Architektur (`marlin.py`) — **nicht angewendet** |

## scripts/

| Datei | Zweck |
|---|---|
| `cleanup_accept.sh <tag>` | Abnahme aller vier Produktionsmodelle gegen die Referenzen |
| `prod_accept.sh`, `prod_accept_p1.sh` | exakter llama-swap-Befehl auf Port 8093 (p1 = nur 27B-MTP, nur Phase 1) |
| `ds_accept.sh <tag>` | DeepSeek PP5, acht Prompts × 2 Läufe (braucht `VENV=`) |
| `dsv4_accept.sh` | DeepSeek-llama-swap-Eintrag, exakter Befehl plus llama-swap |
| `dsv4_ctx_probe.sh`, `dsv4_ctx_cycle.sh` | Kontextsuche DeepSeek (Boot mit ersetzter max-model-len/Blöcken, `EXTRA_ENV=`) |
| `longctx_dsv4_long.py`, `longctx_dsv4.py`, `longctx_dflash2.py` | Langprompt-Proben über Streaming, Decode-Rate ohne TTFT |
| `nccl_sweep.sh <DEVS> <tag> <Varianten…>` | Punkt 13, NCCL-Schalter |
| `prod/…p1.json`, `ds_main.out` | Referenzen, die `cleanup_accept.sh` liest |

## referenz/

Abnahmen vom 10.09. (vor dem Aufräumen): `merge_accept.out` (DFlash2 nach
dem Upstream-Merge, SHA `0106659946c064b1`), `fnq_main_hetero.out` und
`fnq_ref150.out` (Flash-Next neu/alt — Hashes schon untereinander
verschieden), `ds_main.out`, `prod_accept.out`, `dflash2_accept.out`,
`work-main_gegen_origin-main.diff` (Grundlage der Overlay-Inventur).

## ergebnisse/

`cleanup_accept.out` (unterbrochene Abnahme 11.09.),
`flashnext_clean_q3_kuanda.txt` (der q3-Aussetzer), Kontextsuche
(`ctx_1m_512.out` OOM im Profillauf, `ctx_128k_1200.out` bootet,
`ctx_64k2_cycle.out` bestanden), `dsv4_pp5.out` (13k-Test),
`pr_editable_build.log` (Belegbau PR 8).

## scripts/abnahme2/ (Nachmittag/Abend 11.09.)

| Datei | Zweck |
|---|---|
| `driver.sh` | Abnahme-Treiber: 27B-MTP, DFlash2-V100, Flash-Next 3x mit / 3x ohne Aufräumen, DeepSeek; Log `ergebnisse/cleanup_accept2_driver.log` |
| `after.sh` | Nacharbeit-Kette: Chat-Sonde 3x, 11a E2E, E5-A/B, Wheel |
| `flashnext_qual_chat.sh` | Flash-Next-Sonde über den Produktionspfad (Chat-Template, enable_thinking, Parser-Flags) |
| `fnq_check.py` | Vorfilter für Flash-Next-Antworten (Coandă-Regel 11.09., CJK, Wiederholungen) |
| `e2e_11a.sh`, `pr11a_fix.diff` | Paket 11a: main+#572 mit/ohne Fix, RTX-Paar, plus CUDA-Tests auf GPU 4 |
| `e5_ab.sh` | E5-Cache A/B auf dem Produktionsbefehl 27B-MTP (Decode-Rate, SHA, e5-Zeilen) |
| `wheel_test.sh` | Editable-PR: bdist_wheel aus dem Belegbau, Namensvergleich, venv-Kopie, DFlash2-V100 |

`patches/cleanup_committed_43ccb9b8.diff` ist der exakte Diff, der als
`43ccb9b8` committet wurde (Befunde 1, 3–8 zusammen).

Nachtrag 11.09. abends (scripts/abnahme2/):

| Datei | Zweck |
|---|---|
| `rerank_v100.sh` | 11a: die drei SM70-Rerank-CUDA-Fälle auf der echten V100 (PCI-Index 4), mit/ohne Fix — `CUDA_DEVICE_ORDER=PCI_BUS_ID` ist Pflicht |
| `devcap_probe.sh`, `devcap_fix.diff` | Wurzelfix Plattform: Hardware-Probe auf GPU 0+1 (main gegen Fix, beide Geräteordnungen) und der Diff |
| `chat_ask.py`, `after2.sh` | Frageteil der Chat-Sonde gegen einen laufenden Server; Fortsetzungskette nach Sitzungsneustart |
| `ergebnisse/device0_capability_calls_inventory.md` | 263 Capability-Aufrufe ohne Geräteindex in 1Cat-main, grob klassifiziert (Klasse „Modulebene" unzuverlässig) |
| `ergebnisse/e5_ab/` | E5-A/B: Boot-Logs beider Arme, Antworttexte (run1–6), 70,17 gegen 73,11 tok/s |
| `ergebnisse/nacharbeit_after4.log` | Log der Kette: Chat-Sonde 3×3300, 11a E2E (main blockt Turing-NVFP4), E5-A/B |

Lehren: `systemd-run --user` überlebt Sitzungsneustarts, hat aber einen
minimalen PATH (ninja aus dem venv-bin fehlt) und CMake findet darunter die
Python-Header nicht — Bauten aus der Shell per `setsid nohup` starten,
Messungen dürfen in die Unit. Ohne `CUDA_DEVICE_ORDER=PCI_BUS_ID` ist
`CUDA_VISIBLE_DEVICES=4` eine RTX 8000, nicht die V100.

Nachtrag 11.09. nachts: `scripts/abnahme2/driver2.sh` (Abnahme E5-Ausbau +
Befund 2: 27B-MTP, DFlash2-V100, Flash-Next 3× index_share=True, DeepSeek;
Log `ergebnisse/e5out_ishare_accept_driver.log`), `e5_remove.diff` (die zehn
rückwärts angewendeten E5-Hunks), `ergebnisse/flashnext/fnq_acc3_ishare_*`.
`VLLM_SM70_E5_CACHE=0` aus 21 aktiven Skripten entfernt (Archiv unangetastet,
`e5_ab.sh` bleibt als Messvorrichtung mit dem Schalter).
