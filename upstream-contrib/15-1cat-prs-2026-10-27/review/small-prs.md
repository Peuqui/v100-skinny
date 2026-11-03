# Unabhängige Prüfung Loader / MLA-Workspace / Lookahead (27.10. ~12:40)

WERKZEUGFEHLER: `gh pr view --json files` liefert höchstens 100 Dateien; #1260 (threhan) hat 215. Vollständige Listen aller 72 offenen PRs:
scratchpad/prfiles/<N>.txt (gh api --paginate repos/1CatAI/1Cat-vLLM/pulls/N/files), #1260-Patch scratchpad/prfiles/pr1260.patch.
→ ALLE Duplikat-Absätze mit vollständigen Listen neu prüfen.

## Loader-Fixes: READY AFTER FIXES
BLOCKER: LM-Head-Fix (a54a15454) widerspricht #1260: #1260 setzt _layer_execution_policy/_layer_trace in VocabParallelEmbedding.__init__
und testet isinstance(head.quant_method, UnquantizedLinearMethod) für ModelOptNvFp4Config(exclude_modules=["lm_head"]); wir geben None
zurück → Tests schließen sich aus. #1260 ändert auch modelopt.py und bedient ein ModelOpt-NVFP4-GLM-5.3. Text Zeile 9 doppelt falsch.
Option 1: Commit herausnehmen und None-Ansatz als Kommentar an #1260; Option 2: behalten + Absatz „Relation to #1260“ (None behält
UnquantizedEmbeddingMethod = SM70-LM-Head-Provider; #1260 = dichter Provider; unser geänderter Test stammt aus vLLM #42124).
SHOULD-FIX: „found while bringing up GLM-5.3-Flash-FP4-Local on main's loaders“ — main hat kein csf_32x2_v1 → „im Fork gefunden und belegt;
Loader-Code auf main unverändert; CSF kommt mit eigenem PR; auf dieser Basis nicht gebootet“.
NIT: 3b963c896 „38 passed“ vom Fork; Signed-off-by; Pfadzeile 1 (Unterordner-Shards auch für Modelle mit eigenen allow_patterns);
_index_shards = dritter Leser derselben Index-JSON (SSOT).
OK: modelopt.py-Konfliktlösung korrekt; keine Reverts; Absturz auf main real; compressed-tensors-Vergleich; Remote-Download.

## MLA-Workspace: READY AFTER FIXES
BLOCKER: Punkt 3 (15c9746e2, keine kv_b_proj-Profilreservierung bei Sparse-MLA) steckt in #1260 (+ reserve_profile_workspace,
tests/v1/attention/test_mla_profile.py); #1260 ändert auch backends/mla/indexer.py → Punkt 3 + seine zwei Tests (d7c485f6b) herausnehmen
und auf #1260 verweisen, oder ausdrücklich begründen.
SHOULD-FIX: Einheiten Punkt 4: 40 × 1 048 576 × 132 B = 5,16 GiB (Log ab_glm_dflash2_g „Tried to allocate 5.16 GiB“), nicht 5,5; 262 144:
1,29 GiB statt 33 MiB (Text, Commit 77684feca, Kommentar indexer.py:400). Punkte 1/2 helfen nur bei max_num_seqs < 8 und
max_num_seqs × max_model_len < 65 536, nicht „single-sequence server with long context“; mit Punkt 3 nutzt GLM chunked_prefill_workspace_size
gar nicht → GLM-Zahlen für 1/2 sind Zwischenstände, kein dichtes MLA gemessen (ohne Punkt 3 wieder gültig). Pfadzeile 3: „boots at 262k
and 1M“ falsch (1M nur OOM vor dem Fix; Erfolgsboots 262 144 und 524 288); „Unverified scope: none“ → DeepSeek-V3.2 (deepseek_v2.py:666)
unverifiziert, DSv4 am Mini-Checkpoint (mini_ws1.log 387,72 → 10,67 MB).
NIT: „(PP=5 on 32 GiB V100 stages)“ → OOM traf 48-GB-RTX; „Four reservations“ = drei; DCP: Obergrenze auf block_size runden (Assert
mla_attention.py:1529 bei ungeradem max_model_len); Signed-off-by; Wheel-SHA ohne Fork-Commit.
OK: Punkt 4 sicher (Marge genau eine Sequenz); Sparse nur forward_mqa.

## Lookahead: READY AFTER FIXES
SHOULD-FIX: vllm#61589 am 26.10. gemergt (2fb320d83), nicht „open“; unser Port = Stand ~64eca7154 (14.10.); Unterschiede zum Merge:
(a) Hash-Layout, (b) BlockStored.lookahead_token_ids, (c) L = num_spec_tokens für Multi-Modul-MTP (bei uns aus), (d) Streaming keep_end,
(e) Veröffentlichung je Ausgabe → „port of the 2026-10-14 revision; merged in revised form on 2026-10-26“ + Unterschiede, oder neu angleichen.
#1260 „verified MTP prefill cache“ (opt-in, nur Prompt-Blöcke, prefill_boundary.py, find_verified_prefill_cache_hit) → Absatz „Relation to
#1260“. #1265-Beziehung falsch beschrieben (nur speculative.py; disable_eagle_block_drop schaltet Lookahead UND Drop ab). Gemessener Bau
4651800f7 = nur Lookahead-Commit; #1240-Ausrichtung und DSpark-Ausnahme nur im Kontrollboot prodcheck_2026-10-20_lookahead.log → Satz ergänzen.
NIT: f2a8b76d0 mixed_prefill_budget wiederholt #1260-Reparatur; DSpark-Begründung: upstream needs_eagle_block_drop() schließt DFlash/DSpark
aus (Draft-KV hängt nicht vom nächsten Token ab) → zitieren; Signed-off-by fehlt bei 846813365, f2a8b76d0; „upstream #61589“ →
vllm-project/vllm#61589.
OK: drop_eagle_group_ids überall; Hasher-Grenzen; compute_hash unverändert; 24 Tests; keine Reverts; alle Zahlen gegen ab_lookahead belegt.
