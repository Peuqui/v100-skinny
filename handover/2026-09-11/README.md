# Arbeitsdateien der Runde 10./11.09.2026

Gesichert aus dem sitzungsgebundenen Scratchpad. Die Skripte arbeiten
relativ zu ihrem eigenen Ordner (`S=$(dirname "$0")`), laufen also von hier
aus. Übergabe und Reihenfolge: `../../HANDOVER.md`.

## patches/ (gegen den Worktree `1Cat-vLLM-work`)

| Datei | Stand |
|---|---|
| `cleanup_neutral.diff` | Befunde 1, 3, 4, 5, 7, 8 — **angewendet**, unkommittiert |
| `cleanup_befund6.diff` | Befund 6 (`kv_cache_utils.py`) — **angewendet**, unkommittiert |
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
