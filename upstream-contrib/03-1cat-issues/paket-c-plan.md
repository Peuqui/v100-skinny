# Paket C — sm75-FA2 (Turing FlashAttention-2) für 1Cat (Stand 12.09. nachmittags)

## Ausgangslage (belegt)

- 1Cat baut FA2 nur für ≥ 8.0 (vllm-project/flash-attention @ bce29425) oder
  für 7.0 (zhinianqin/flash-attention-v100 @ c2eda5e6 + vier `cmake/patches/
  sm70_*.patch`, dazu eigene `sm70_v37`/`grouped_long`-Quellen als
  `target_sources` in `_vllm_fa2_C`). Für 7.5 gibt es nichts
  (`CMakeLists.txt:1575-1590`).
- Backend-Wahl auf Turing (`platforms/cuda.py`): allgemeiner Zweig
  `FLASH_ATTN, FLASHINFER, TRITON_ATTN, …` — FLASH_ATTN scheitert ohne
  Bibliothek, FlashInfer stürzt in `BatchPrefillWithPagedKVCache` (invalid
  argument), Triton läuft (kausal only, kein fp8-KV-Cast auf sm75).
  `FLASH_ATTN_V100` prüft hart `== (7, 0)`.
- Unser Fork: `Peuqui/flash-attention` Branch `sm75-enablement-pr` @ 6da42e8
  (gepusht), auf Upstream-Pin 28e862d + 6 Upstream-Commits. Eigener Code:
  **7 Dateien, 232+/31−** (`CMakeLists.txt` FA2_ARCHS "7.5;8.0+PTX", BF16 für
  7.5 raus; `flash_api.cpp`, `flash_fwd_kernel.h`, `flash_fwd_launch_template.h`,
  `static_switch.h`, `utils.h` (kMmaPerCopy-Fix, latenter Upstream-Bug bei
  hdim > 64 auf sm75), `flash_attn_interface.py`). Zwei Commits: „Enable and
  fix the FA2 forward path for Turing (sm75)", „Enable split-KV for paged
  multi-token queries (spec-decode verify)".
- Produktiver Drop-in: `_vllm_fa2_C_sm75.abi3.so` (167 MB, md5 1285b8f0e013,
  Bau 03.09.) neben 1Cats `_vllm_fa2_C.abi3.so`; Lader pro Gerät im Fork
  (`flash_attn_interface.py` +83, `cuda.py` +19 (Wahl (7,5) → FLASH_ATTN
  zuerst, Capability vom eigenen Gerät), `flash_attn.py` +7 (≥ (7,5),
  fp16-only-Gate)). Seit 06.09. in Produktion auf der RTX — mit **fp16-KV**:
  der Fork ignoriert unter SM80 die FP8-KV-Vorgabe des Checkpoints
  (`checkpoint_kv_quant_allowed`, Overlay, nicht eingereicht); der
  Produktions-Boot 11.09. 13:01 hat keine "Using fp8"-Zeile. Die frühere
  Fassung dieses Plans ("fp8-KV läuft") war falsch.
- **FA2 + fp8-KV ist per Konstruktion unmöglich** (Messung 12.09. 20:00):
  `FlashAttentionBackend.supports_kv_cache_dtype` erlaubt fp8 nur mit FA3
  auf 9.x; Main und Fork identisch. Auf reinem Main löst `auto` beim
  ModelOpt-NVFP4-27B zu `fp8_e4m3` auf (Checkpoint `kv_cache_quant_algo: FP8`)
  → FLASH_ATTN abgelehnt → TRITON_ATTN → fp8-Cast auf sm75 stürzt. Ohne
  `--kv-cache-dtype float16` bootet so ein Checkpoint auf Turing gar nicht;
  Paket C allein löst das nicht. Der vierte Schritt der Reihenfolge
  ("fp8-KV-Cast-Fix") ist damit eher die Overlay-Politik
  `checkpoint_kv_quant_allowed` als ein Triton-Cast.
- **Bauform-Fund:** unser FA-Fork nimmt den Interpreter als
  `Python_EXECUTABLE` (nicht `VLLM_PYTHON_EXECUTABLE` wie 1Cat), sonst
  Configure-Fehler "No module named torch". Der ExternalProject muss die
  Variable übersetzen.

## Bauform-Frage (Entscheidung Peuqui)

Zwei FA2-Bibliotheken müssen nebeneinander existieren (V100 + Turing im
selben Wheel/Baum); beide Projekte heißen `_vllm_fa2_C`, und
`vllm_flash_attn.cmake` warnt, dass das Projekt globale CMake-Funktionen
überschreibt — zwei FetchContent-Kopien kollidieren.
- **Vorschlag: `ExternalProject_Add`** für die sm75-Bibliothek (eigener
  Konfigurier-/Bauprozess, keine Funktionskollision), nur wenn 7.5 ∈
  CUDA_ARCHS: Quelle `Peuqui/flash-attention` @ Pin (Tag setzen),
  `VLLM_GPU_ARCHES=75-real`, FA2 an/FA3 aus, fp16-only, forward-only,
  Extension-Name `_vllm_fa2_C_sm75`, Installation nach
  `vllm/vllm_flash_attn/`. Dafür braucht unser FA-Fork eine kleine
  CMake-Option für den Extension-Namen (Standard `_vllm_fa2_C`, keine
  Änderung für alle anderen).
- Alternative: FA-sm75-Änderungen zuerst upstream (vllm-project/flash-
  attention) — von Peuqui eingefroren (11.09.); 1Cat pinnt ohnehin schon
  zhinianqins Fork, ein weiterer gepinnter Fork ist dort üblich.

## PR-Zuschnitt

- **C-0 (FA-Fork, klein):** CMake-Option `VLLM_FA2_OUTPUT_NAME` (umgesetzt, uncommitted); Tag.
- **C-1 (1Cat, CMake):** ExternalProject für 7.5; nichts an der V100-Spur.
- **C-2 (1Cat, Python):** Lader pro Gerät, Backend-Wahl (7,5) → FLASH_ATTN
  (behebt damit auch den FlashInfer-Absturz als Default), Capability ≥ (7,5)
  mit fp16-Gate. Ggf. als eigener, vorgezogener Dreizeiler: FlashInfer auf
  sm75 nicht anbieten.
- **Belege:** RTX 27B (E-1-Pfad, fp8-KV wie Produktion): FA2 gegen Triton —
  kurz (400 Token), 13k-Vorkontext TTFT und Decode; V100 unverändert; Tests
  aus dem FA-Fork (`tests/sm75_debug_*.py`, JIT-Proben) als Kernel-Belege;
  Duplikatssuche ("sm75", "turing", "flash attention"). Messkette läuft:
  `handover/2026-09-12/fa2_chain.sh` (fa2_fp8 / tri_fp16 / fa2_fp16);
  fa2_fp8 scheidet aus (s. o.), Vergleich ist tri_fp16 gegen fa2_fp16.
  Eigenständiger sm75-Bau aus dem Fork mit `VLLM_FA2_OUTPUT_NAME`:
  `handover/2026-09-12/fa2_build_sm75.sh` (Nachweis für C-0/C-1).

## C-1 im Detail (Entwurf 12.09. abends, nach Lesen von setup.py und vllm_flash_attn.cmake)

- 1Cat entscheidet die FA-Quelle einmalig: 8.0+ → vllm-project @ bce29425,
  sonst 7.0 → zhinianqin @ c2eda5e6 + 4 Patches; für "7.0;7.5" kommt nur
  die V100-Spur, für "7.5;8.0" nur Ampere-Kernel. 7.5 bleibt in jedem Fall
  ohne FA2.
- Präzedenzfälle im Baum: (a) Git-Pin eines Fremd-Forks per FetchContent
  (zhinianqin), (b) vendored Quellbaum `flash-attention-v100/` (30 Dateien,
  eigenes Paket, per `bundle_flash_attn_v100` in setup.py gebaut und ins
  Wheel gelegt), (c) JIT-Vorbau `bundle_flash_qla_sm70`. Für die volle
  FA2-Quelle (csrc + cutlass-Submodul) ist nur (a) tragbar.
- Zwei FetchContent-Kopien desselben Projekts kollidieren (globale
  CMake-Funktionen, gleicher Target-Name `_vllm_fa2_C`). Daher
  `ExternalProject_Add(vllm-flash-attn-sm75)` in einer neuen Datei
  `cmake/external_projects/vllm_flash_attn_sm75.cmake`, eingebunden wenn
  7.5 ∈ CUDA_ARCHS, unabhängig von der bestehenden Entscheidung:
  GIT_REPOSITORY Peuqui/flash-attention, GIT_TAG <Pin>, CMAKE_ARGS
  `-DPython_EXECUTABLE=${VLLM_PYTHON_EXECUTABLE}` (Namensübersetzung!),
  `-DCUDA_ARCHS=7.5`, `-DFA2_ENABLED=ON -DFA3_ENABLED=OFF`,
  `-DVLLM_FA2_OUTPUT_NAME=_vllm_fa2_C_sm75`, gleiche CMAKE_BUILD_TYPE /
  CMAKE_CUDA_COMPILER; BUILD_BYPRODUCTS `<BINARY_DIR>/_vllm_fa2_C_sm75.abi3.so`;
  `install(FILES … DESTINATION vllm/vllm_flash_attn COMPONENT _vllm_fa2_C_sm75)`.
- setup.py: `ext_modules.append(CMakeExtension("vllm.vllm_flash_attn._vllm_fa2_C_sm75"))`
  wenn `_cuda_arch_contains(7, 5)` — build_extensions baut per
  `--target=<Name>` und installiert per `--component <Name>`, also muss der
  ExternalProject-Target genau `_vllm_fa2_C_sm75` heißen. Dazu
  `exact_members` der Precompiled-Wheel-Entnahme um die neue .so ergänzen.
- Bauzeit-Nachweis: `fa2_build_sm75.sh` (eigenständig, gleiche Argumente).

## Entscheidungen und Messstand (12.09. abends)

- **FA2 mit fp8-KV auf Turing: GESTRICHEN (Peuqui, 12.09.).** Wäre ein
  zweites Kernelprojekt (8-Bit-Kacheln, Register-Konvertierung ohne
  Hardware-cvt, neue C++-API mit Skalen) ohne Nutzen: KV bei 262k in fp16
  27B 17 GB (16/64 Attention-Schichten, 4 KV-Köpfe), Flash-Next 6,4 GB
  (12/48, 2 KV-Köpfe), DeepSeek läuft über MLA. Turing rechnet in fp16 am
  schnellsten. Vierter Schritt der Reihenfolge ist die Overlay-Politik
  `checkpoint_kv_quant_allowed` (Boot unter `auto` auf Turing).
- **Kurz-Sonde (400 Token, RTX-Paar, MTP k=3, fp16-KV):** Triton 69,85
  gegen FA2 73,04 tok/s, Text-SHA identisch (38848c08a44405ae, wie E-1).
- **13k-Sonde, erster Lauf UNGÜLTIG:** Rohprompt ohne Chat-Template, der
  Kontext-Text kam leer zurück (SHA des leeren Strings), die Decode-Raten
  (6,34 gegen 22,38 tok/s) sind damit Artefakte aus zu wenigen Token.
  Zweitlauf `fa2_chain2.sh`/`fa2_probe2.sh`: Chat-Endpunkt, Denken aus,
  Tokenzahl/finish_reason/Text protokolliert (`ctx_response.json`).

## Ergebnis Zweitmessung und Veröffentlichung (12.09. 21:00)

| RTX-Paar, 27B-NVFP4, TP2, MTP k=3, greedy, fp16-KV, Main ae75fb9b + E-1 + E-2 | TRITON_ATTN | FLASH_ATTN sm75 |
|---|---|---|
| Kurz-Sonde 400 Token (Median aus 5) | 70,91 tok/s | 74,19 tok/s |
| 13k Vorkontext (13012 Token), TTFT max_tokens=1 (Median aus 3) | 36,26 s | 17,75 s |
| 13k Vorkontext, Decode 200 Token (Median aus 3) | 15,03 tok/s | 56,48 tok/s |

Beide Läufe: 200/200/200 Token, finish_reason length, Kurz-SHA 38848c08a44405ae
und 13k-SHA 874dc1b410ba4a2e jeweils identisch. FA2-Lauf mit der FRISCH
gebauten Bibliothek (md5 70217c57b187, 38 Kernel sm_75, kein PTX) aus
`fa2_build_sm75.sh`, nicht mit dem Drop-in vom 03.09. (der liegt als
`.drop-in-0309` daneben im Worktree `1Cat-vLLM-pr-turing-ops`).
Rohdaten: `~/.cache/mtp-diagnostics/fa2_{tri_fp16,fa2_fp16}/{result.json,ctx_response.json,boot.log}`.

**1Cat-Issue #612 veröffentlicht** (Freigabe Peuqui, Text ohne Empfehlung,
drei Formen zur Wahl): https://github.com/1CatAI/1Cat-vLLM/issues/612.
Nächster Schritt hängt an der Antwort: Form 1/2 → CMake-Option im FA-Fork
committen + Tag, ExternalProject in `cmake/external_projects/
vllm_flash_attn_sm75.cmake`, setup.py-Extension, dann C-2 aus dem Worktree
`1Cat-vLLM-pr-fa2sm75` (Branch `sm75-fa2-pr`, drei Python-Diffs, Lint+mypy
grün). Form 3 → Vendoring-Plan neu schneiden. Keine Antwort → Form 1 als PR
einreichen (Ansage Peuqui 12.09.: „wenn er nicht antwortet, reichen wir ein").
