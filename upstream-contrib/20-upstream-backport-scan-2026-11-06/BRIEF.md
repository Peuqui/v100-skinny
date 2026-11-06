# Prüfauftrag: Upstream-vLLM-Bugfixes als Backport-Kandidaten für unseren 1Cat-Fork

## Hintergrund
- Unser Fork (github.com/Peuqui/1Cat-vLLM, Zweig `fork-main-v6`) basiert auf 1Cat-vLLM. 1Cat hat seine vLLM-Basis am
  29.05.2026 eingefroren (letzter Upstream-Commit vllm#43616) und zieht upstream nicht nach; Upstream-Fixes kommen nur als
  einzelne Backports. Alles, was upstream seit dem 29.05. behoben wurde, fehlt also bei uns, außer 1Cat oder wir haben es
  selbst (oft anders) gelöst.
- Arbeitsbaum (NUR LESEN): `/home/mp/Projekte/vllm-research/1Cat-vLLM-main-v6` (= was in Produktion läuft, plus der Port vllm#60472).
- Pro Kandidat liegt der Upstream-Patch (Merge-Commit) unter `/home/mp/.cache/upstream-backport-scan-2026-11-06/patches/<nr>.patch`,
  die PR-Beschreibung (gekürzt auf 4000 Zeichen) in `candidate_files.json` unter dem Schlüssel `"<nr>"` → `body`. Mehr bei Bedarf:
  `gh pr view <nr> --repo vllm-project/vllm --comments`; Upstream-Datei zum Merge-Stand:
  `gh api -H "Accept: application/vnd.github.raw" "repos/vllm-project/vllm/contents/<pfad>?ref=<merge-sha>"`.
- Status je Kandidat (maschinell, nur für die Dateien, die es bei uns gibt): `fehlt_passt` = der Upstream-Patch lässt sich auf
  unseren Baum anwenden (Fix fehlt, direkt übernehmbar); `teilweise` = nur einige Dateien passen (`per_file`); `konflikt` = passt
  nicht: Code verschoben/umgebaut, von 1Cat anders gelöst, oder der fehlerhafte Code existiert bei uns gar nicht. Bei `konflikt` ist
  die Frage also immer: Gibt es den Fehler in UNSEREM Code (ggf. an anderer Stelle oder in 1Cats Fassung)?

## Unser Betrieb (nur das zählt für „trifft uns“)
- Karten: 3× Tesla V100 (SM70) + 2× Quadro RTX 8000 (SM75). KEIN Hopper/Blackwell: FlashInfer, FlashMLA, DeepGEMM, TRT-LLM,
  CUTLASS-SM90+-Pfade laufen bei uns nicht. 1Cat hat eigene SM70/SM75-Kerne (TurboMind, QPN8-FP8, flash_v100-Backend, eigener
  Triton „sm7x“, eigene Sparse-MLA-/Indexer-Kerne). FA2 nur für den DFlash2-Entwurf auf Turing.
- Engine: V1-Engine, Model Runner V2 (`VLLM_USE_V2_MODEL_RUNNER=1`), Multiproc-Executor (kein Ray), kein DP, kein EP, kein LoRA,
  keine KV-Connectoren/-Offload, kein Pooling, kein Sleep-Mode, kein Beam Search, keine Structured Outputs/Grammatiken (soweit
  bekannt), CUDA-Graphs + torch.compile an, Präfix-Cache an, Chunked Prefill (`--max-num-batched-tokens 1024`, kurze Prompts in
  512er-Stücken), async scheduling Standard.
- Modelle in Produktion:
  - DeepSeek-V4-Flash 284B (`DeepseekV4ForCausalLM`, MXFP4/FP8), PP5 über alle fünf Karten, max_num_seqs 1, KV fp8, 500k Kontext,
    spekulativ `dspark` (k=5, Coding-Eintrag k=7), Tool-Parser `deepseek_v4`, Reasoning-Parser `deepseek_v4`. 1Cats Modellpaket
    `vllm/models/deepseek_v4/`, DSpark-Kopf `vllm/v1/worker/gpu/spec_decode/dspark/`.
  - GLM-5.3-Flash NVFP4 (`Glm5NextForConditionalGeneration`, Paket `vllm/models/glm5next/`, Sparse-MLA mit Indexer/„kpool“,
    MTP-Checkpoint), PP5, max_num_seqs 1, KV fp8_e4m3, 524k Kontext, Bildteil an (bis 12 Bilder, kein Video), spekulativ `dflash`
    mit einem DFlash2-Entwurf (`qwen3_dflash2.py`, Block-FP8, k=7), Tool-Parser `glm47`, Reasoning-Parser `deepseek_r1`.
  - Qwen3.8-27B FP8 (`Qwen3_5ForConditionalGeneration`, `qwen3_5.py`/`qwen3_5_mtp.py`, Gated-DeltaNet), TP2 auf den RTX,
    max_num_seqs 3, spekulativ `mtp` (k=3), Bilder bis 12, Tool-Parser `qwen3_coder`, Reasoning-Parser `qwen3`.
- Frontend: OpenAI `/v1/chat/completions` mit Streaming und Tools (Client: AIfred), `--enable-prompt-tokens-details`,
  `enable_thinking`-Chat-Template-Kwargs; Bilder als URL/base64. Keine Responses-/Anthropic-API, kein /v1/completions-Echo.

## Aufgabe für deinen Bereich
1. Sichte ALLE Kandidaten deiner Liste zuerst nach Titel und Beschreibung. Sortiere sofort aus, was uns sicher nicht trifft
   (anderes Modell, andere Plattform, Funktion, die wir nicht nutzen), mit Stichwort als Grund.
2. Für die übrigen: lies Patch und unseren Code und beantworte (a) Kann der Fehler in unserem Betrieb auftreten? (b) Ist er bei uns
   schon gelöst (von 1Cat oder uns, ggf. anders) — mit Datei:Zeile als Beleg? (c) Empfehlung: `übernehmen` (Diff passt), `anpassen`
   (Fix gilt, Code weicht ab: kurz sagen wie), `nein` (Grund). (d) Kann er Tempo oder Ausgabe ändern (dann braucht es eine A/B)?
   (e) Abhängigkeiten zu anderen Kandidaten (Reihenfolge, Vorgänger-PR, „Revert“-PRs beachten: ein später revertierter PR ist kein
   Kandidat).
3. Keine Vermutungen als Befund ausgeben: Wo du es nicht belegen kannst, schreib `unklar` und was zur Klärung fehlt.

## Regeln
- NUR LESEN: keine Änderungen an Git-Bäumen (kein checkout, kein apply ohne `--check`, keine Commits), keine Builds, keine Tests,
  keine GPU, kein pip. `git -C <baum> apply --check [-R] --include=<datei> <patch>` ist erlaubt.
- Schreibe dein Ergebnis auf Deutsch nach `/home/mp/.cache/upstream-backport-scan-2026-11-06/review_<bereich>.md`:
  oben eine Rangliste der Empfehlungen `übernehmen`/`anpassen` (wichtigste zuerst: falsche Ausgabe, Hänger/Abstürze, Speicher,
  dann Rest), je Zeile: PR-Nummer, Kurztitel, warum es uns trifft (1 Satz), Beleg im Code, Empfehlung, A/B nötig ja/nein,
  Abhängigkeiten. Darunter die Tabelle aller geprüften Kandidaten mit Kurzurteil, ganz unten die Aussortierten in einer Zeile je PR.
- Deine Antwort an mich: höchstens 30 Zeilen, nur die Rangliste (übernehmen/anpassen) und wie viele du aussortiert hast.
