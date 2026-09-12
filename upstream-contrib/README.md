# Upstream-Beiträge

**Stand 2026-09-07.** Entwürfe und Diffs liegen in den Unterordnern; der
laufende Status der 1Cat-Einreichungen steht unten. Regeln für neue Beiträge:
`AGENTS.md` im 1Cat-Repo (Duplikatsprüfung, Testkommandos samt Ergebnis im
PR-Text, KI-Einsatz deklarieren) — Verstoß kann eine Sperre nach sich ziehen.
**Nichts senden ohne Freigabe von Peuqui.**

## 1Cat-vLLM: gemergte PRs (Stand 2026-09-07, per `gh pr list` geprüft)

| PR | Titel |
|---|---|
| #469 | [Bugfix][Perf][SM70/SM75] Widen QSA sparse launch profile to pre-Ampere |
| #485 | [Bugfix][SM70] Skip final mixer weights on non-last PP ranks |
| #511 | [Bugfix][Spec Decode] Bind self.drafter on non-last PP ranks (#439) |
| #512 | [Bugfix][Spec Decode] Report SM70 MTP profiles from the last PP stage (#414) |
| #514 | [Bugfix][SM70] Gate SM70 config defaults on any visible device (#412) |
| #516 | [Core][Qwen4Exp] Gate PLE under PP on the partition, not the PP size (#479) |
| #518 | [Core] Profile the KV budget on a warm forward, not the cold torch.compile |
| #528 | [Model][Qwen4Exp] Split the pinned-host PLE table between device and host (#479) |
| #536 | [Bugfix] Hash unregistered VLLM_ env vars into the compile cache key |

Geschlossen ohne Merge: #455 (Pre-Ampere-Tile-Profile) — auf dem eigenen Fork
statt gegen Upstream-main verifiziert, mit Eingeständnis zurückgezogen. Daraus
die Regel, jede Behauptung vor dem Senden gegen frisch gefetchten Upstream zu
prüfen.

Damit sind neun PRs gemergt; die Repo-Regel „Autor braucht ≥4 gemergte PRs"
(roter `pre-run-check`) ist erfüllt.

## 1Cat-vLLM: offene PRs (Stand 2026-09-11 abends)

| PR | Titel | Entwurf |
|---|---|---|
| #572 | [Bugfix][Perf][SM70] Make Turing (sm75) boot, compute correctly and keep its Inductor fusions | `03-1cat-issues/pr-turing-four-fixes.md` |
| #573 | [Bugfix][Qwen4Exp] Keep the MTP drafter stage-local under pipeline parallelism | `pr-qwen4exp-mtp-pp.md` |
| #574 | [Bugfix][Spec Decode] Trim the optimistic spec-decode tokens on every pipeline rank | — |
| #576 | [Bugfix][SM70] Read the quantization SM70 gate from the worker's own device | — |
| #592 | [Bugfix] DFlash: fuse context K/V through quant_method so a quantized draft head loads | `pr-dflash-quantized-draft-context-kv.md` |
| #599 | [Bugfix][Spec Decode][SM70] Gate DFlash2's BF16 emulation and FlashInfer top-k on the worker's own device | `pr-dflash2-pre-sm80-worker-device.md` |
| #600 | [Bugfix][Platform] Resolve an unspecified device_id to the worker's own device | `pr-platform-default-device-id.md` |
| #601 | [Bugfix][Build][SM70] Declare the pybind11 SM70 extensions non-limited-API | `pr-editable-soabi-modules.md` |
| #603 | [Bugfix][DeepSeek-V4] Align the SWA decode threshold with the sparse MLA builder | `pr-sparse-swa-spec-threshold.md` |
| #604 | [Feature][SM75] Run ModelOpt NVFP4 and FP8 linears on Turing through the SM70 QPN kernels | `pr-turing-nvfp4-fp8-linear.md` (Paket E-1) |

#599–#601 eröffnet 2026-09-11 abends (Freigabe Peuqui). #603 eröffnet 2026-09-12 früh (Freigabe Peuqui); #572 und #573 am 2026-09-12 03:30 GEMERGT — Overlay-Teile beim nächsten main-Hereinholen entfernen. #600 ist der
Wurzelfix für die Gerät-0-Fehlerklasse aus #412 und macht die Einzelfixes
#514/#576/#599 überflüssig, nicht falsch. Nach dem Merge eines eigenen PRs
den zugehörigen Overlay-Teil im Produktions-Worktree ENTFERNEN (STAND.md).

---

## Erste Runde — alle veröffentlicht am 2026-08-28

ALLE VERÖFFENTLICHT am 2026-08-28 (Freigabe Peuqui):
- PR:  https://github.com/dnv2003/v100-skinny/pull/7
- vLLM: https://github.com/vllm-project/vllm/issues/54260 (Bug in main
  bestätigt, Zeile 1425, Stand 28.08.)
- 1Cat: https://github.com/1CatAI/1Cat-vLLM/issues/412 (Device-0),
  /413 (E5×QSA), /414 (MTP-Profiling PP)
- HF:  https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/discussions/6
- Werkzeug-Repo: https://github.com/Peuqui/mtp-quant-transplant

| # | Ziel | Art | Inhalt | Voraussetzung |
|---|------|-----|--------|---------------|
| 1 | vllm-project/vllm | Issue (+PR-Angebot) | PP+async+spec: Output-Trim läuft nur auf letzter Stufe (elif→if) | gegen aktuellen main verifizieren |
| 2 | dnv2003/v100-skinny | PR | Branch pp-mtp-merge (PP×TP+MTP, sm75-Paket, Device-0-Fixes, PLE-Kaskade) | GitHub-Fork unter Peuquis Account, Branch pushen |
| 3 | 1CatAI/1Cat-vLLM | Issue 1 | Capability-Gates fragen Device 0 statt aller sichtbaren GPUs | — |
| 4 | 1CatAI/1Cat-vLLM | Issue 2 | E5-Metadaten-Cache crasht an CSA/QSA-Modellen (shape [] vs [1]) | — |
| 5 | 1CatAI/1Cat-vLLM | Issue 3 | MTP-Profiling-Report bei PP blind (is_global_first_rank-Gate) | — |
| 6 | HF RadixArk/Qwen3.8-Flash-Next-NVFP4 | Discussion | Unquantisierter MTP-Block = Spekulation wird Verlustgeschäft auf Pre-Hopper | — |

Messgrundlage: docs/journal/QWEN4EXP-PORT-HANDOVER.md (Abschnitte 28.08.) und
docs/journal/MERGE-PROJECT-HANDOVER.md. Hardware: 2x Quadro RTX 8000 (sm75) + 3x
Tesla V100 (sm70), TP=2/PP=2, 1Cat-vLLM 1.3.0 + v100-skinny-Patches.
