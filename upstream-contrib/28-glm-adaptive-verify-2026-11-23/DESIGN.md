# Variable Prüflänge für GLM-5.3-Flash (DFlash2): Entwurf

Stand 23.11.2026, Baum `1Cat-vLLM-merge-v14` (Zweig `merge-1cat-2d218ab03`), nur gelesen, nichts gebaut, nichts gemessen.
Pfade relativ zum Baum. Kennzeichnung: **[belegt]** = im Quelltext gelesen (Datei:Zeile), **[abgeleitet]** = Schluss aus dem Code oder aus
den gelieferten Messzahlen, nicht ausgeführt, **[offen]** = muss gemessen oder getestet werden.

## 0. Ergebnis in Kürze

1. **Architektur A (empfohlen):** Der Entwurf läuft unverändert mit seinem vollen Block K_max (Konfiguration `num_speculative_tokens` = K_max,
   Block 1 + K_max wie früher bei k=7), nur der **Scheduler** bestimmt je Runde n ≤ K_max und plant 1 + n Token ein. Der Worker, der
   Rejection-Sampler, die PP-Pakete und der Entwurf brauchen **keine** Änderung, weil sie die Anzahl der Entwurfstoken schon heute je
   Anfrage aus dem Scheduler-Output lesen (Belege in Abschnitt 1). Neu nötig sind: (a) der Regler im Scheduler für Methode `dflash`,
   (b) **Tail-Graphen** der Zielmodell-Prüfrunde für 1 + n Zeilen auf jeder PP-Stufe (sonst wird auf K_max + 1 aufgefüllt und es wird
   nichts gespart), (c) Konfigurationsfelder, (d) Tests und Messungen.
2. **Architektur B (Entwurfsblock je Runde variabel)** ist deutlich aufwendiger (Faltung mit fest eingebackener Blockgröße, Entwurfsgraphen
   je Blockgröße, Eingabekernel, Selektor) und spart nur die Entwurfskosten zwischen Block 1 + n und Block 1 + K_max, die
   wahrscheinlich klein sind **[abgeleitet]**. Nicht empfohlen; erst wenn eine Messung zeigt, dass der Entwurf je Block-Zeile mehr als
   etwa 1 ms kostet.
3. **Regler:** keine Bänder, sondern direkt `argmax_n  E(n) / (t0 + c·n)` aus je Position geschätzter kumulativer Annahme A_j
   (Abschnitt 5). Einziger Kalibrierwert ist das Kostenverhältnis ρ = c / t0 ≈ 0,108 (aus 4,2 ms / 39 ms). Mit den gelieferten
   Positionswerten ergibt die Regel n = 3 (AIfred-Sampling), n = 4 (greedy, Plateau 3..5), n = 7 (Rechnen mit hoher Annahme); das
   deckt sich in der Richtung mit den Messungen je Textsorte.
4. **Kosten von K_max = 7:** KV-Pool 550 502 statt 655 360 Token (gemessen, STAND.md Nachtrag 23.11.). Deshalb K_max wählbar halten;
   Vorschlag für die erste Messung K_max = 6, endgültig per A/B.

## 1. Belegte Ausgangslage (Code)

### 1.1 Scheduler, Platzhalter, Async

- `vllm/v1/core/sched/async_scheduler.py:19` `_spec_token_placeholders = (-1,) * num_spec_tokens`. `:118-162` `_update_after_schedule`:
  `:134` `next_k = self._select_adaptive_batch_k(decode_req_ids)`, `:137-140` Platzhalter-Tupel der Länge next_k,
  `:153` `num_output_placeholders += 1 + len(scheduled_spec_decode_tokens[req])` (tatsächlich geplante Anzahl), `:156-159`
  `request.spec_token_ids = list(placeholders)` (nur wenn `num_computed_tokens + next_k <= drafter_max_model_len`), `:162`
  `next_decode_eligible_step = current_step + pp_size`.
  **Die Platzhalterlänge dieser Planrunde ist die Prüflänge der nächsten Runde derselben Anfrage.**
- `vllm/v1/core/sched/scheduler.py:1385-1432`: geplante Entwurfstoken = `request.spec_token_ids[:num_scheduled_spec_tokens]`, wobei
  `num_scheduled_spec_tokens` aus `num_new_tokens` folgt (`Request.num_tokens_with_spec` = alle Token + `len(spec_token_ids)`,
  `vllm/v1/request.py:284-285`). Danach `request.spec_token_ids = []` (`:1432`), neue Platzhalter setzt der AsyncScheduler.
- `scheduler.py:2604-2629` (`update_from_output`): `num_draft_tokens = len(scheduled_spec_decode_tokens[req])`, `num_accepted =
  len(generated) - 1`, Rücknahme von `num_computed_tokens` und `num_output_placeholders` um `num_draft - num_accepted`, danach der
  Haken `self._on_spec_decode_feedback(req_id, num_draft_tokens, num_accepted)`; Basisklasse leer (`:3397-3403`). Der Haken läuft nur,
  wenn `scheduled_spec_token_ids and generated_token_ids` (also nicht bei Runden ohne Entwurf).
- `scheduler.py:3029-3105` `update_draft_token_ids_in_output` kürzt Entwurfszeilen auf die Platzhalterlänge (nur Strukturausgaben / Sync-Pfad).
- `scheduler.py:284-285` und `vllm/config/vllm.py:683-690`: `num_lookahead_tokens` für DFlash = `num_speculative_state_tokens() + 1`
  (= K_max + 1, fest). Blockzuteilung ist also für K_max gerechnet, unabhängig von n.

### 1.2 1Cats adaptiver MTP (EntMTP)

- `vllm/config/kernel.py:1727-1758` `is_adaptive_mtp_enabled`: verlangt `method == "mtp"` und `num_spec_tokens >= 3`; `auto` nur bei
  `num_spec_tokens >= 5` und `is_device_capability((7, 0))` des Scheduler-Prozesses. `:1761-1775` `adaptive_mtp_min_k` (Standard 2).
  Felder in `vllm/config/speculative_sampling.py:76-79` (`adaptive_mtp`, `adaptive_mtp_min_k`), Alias auf `VLLM_1CAT_ADAPTIVE_MTP*`
  (`:86-87`).
- `async_scheduler.py:33-45` Zustand je Anfrage `[ema_successes, ema_trials, cur_k]`; `:47-100` Regler: alpha 0,28, p̂ = Σ/Σ mit
  Versuchen `min(accepted + 1, num_draft)`, feste Bänder 0,62/0,74/0,82/0,89 → k = 2/3/4/5/6, Hysterese 0,025, sofortiges +1
  bei voller Annahme. `:102-116` nur bei genau einer decodierenden Anfrage, sonst `num_spec_tokens`.
- `async_scheduler.py:135-136` schreibt `scheduler_output.next_spec_steps` (`vllm/v1/core/sched/output.py:303`); Worker liest es
  `vllm/v1/worker/gpu/model_runner.py:1885,1948-1949` und gibt `active_spec_steps` nur an Sprecher mit Attribut `multi_step_decode`
  weiter (`:2072-2075`); nur `eagle/speculator.py:673-678,966,1112-1120` kürzt damit den Entwurf. DFlash/DFlash2 haben kein
  `multi_step_decode` **[belegt, per grep]**.
- Tail-Graphen: `vllm/v1/worker/gpu/cudagraph_utils.py:256-266` (`_adaptive_mtp_tail_graphs`, Längen `decode_query_len-1 … 3`),
  Zulassung nur Einzelanfrage-Tails `:340-352`.

### 1.3 Was bei DFlash2 heute schon variable Längen verträgt (Beleg, dass A wenig Worker-Arbeit braucht)

- Eingabeaufbereitung je Anfrage: `model_runner.py:1199-1235` liest `len(scheduled_spec_decode_tokens[req])`, baut `cu_num_logits`,
  `expanded_idx_mapping` mit `max_expand_len = num_speculative_steps + 1`. `input_batch.py:373-408`, Kernel `:311-366`: schreibt je
  Anfrage nur `num_logits - 1` Entwurfstoken aus `req_states.draft_tokens` (Breite K_max) nach `input_ids`. **Der Entwurf muss also nicht
  gekürzt werden; Spalten hinter n werden nie gelesen.**
- Rücknahme `get_num_sampled_and_rejected` (`input_batch.py:411-457`): `num_rejected = num_logits - num_sampled` je Anfrage.
- Rejection-Sampler dicht: `spec_decode/rejection_sampler.py:102-148` nutzt `cu_num_logits`, `num_speculative_steps` nur als
  Maximalbreite (`rejection_sampler_utils.py:82-89`: `draft_step_idx >= num_speculative_steps` überspringen). Greedy-Prüfung
  `sm70_greedy_verify.py:22-62` ebenso über `cu_num_logits`.
- SM70-Sparse-Pfad `dflash2/sparse_rejection.py:543-733` und Sampler-Graph `dflash2/sampler_graph.py:25,449-454,529` gelten nur auf
  Compute-Capability (7,0) bzw. TP4. **Die letzte PP-Stufe von GLM ist eine RTX 8000 (7,5), TP1**: beide Pfade sind dort nicht aktiv
  (`sparse_rejection.py:566-567`, `sampler_graph.py:529`) **[abgeleitet aus Topologie; per Boot-Log bestätigen]**. Es läuft der dichte Pfad.
- PP-Übertragung `pp_utils.py:97-100,130-157,276-316`: Paketbreite `max_sample_len = K_max+1`, `max_draft_len = K_max`, fest. Antwortzeilen
  mit weniger Token sind mit -1 aufgefüllt (`:49-55`), die Entwurfsbreite ist immer volle Breite K_max (Entwurf ist unabhängig von n).
  **Keine Änderung nötig.** Die Zwischentensoren zwischen den Stufen kopieren `[:n]` Zeilen (`model_runner.py:1780-1782`).
- Entwurf (DFlash2) liest die Zielzeilen der **tatsächlichen** Runde: `dflash/speculator.py:558-600` (`num_target_tokens =
  input_batch.num_tokens`, Puffer „so groß wie das Ziel“), Kernel `_prepare_dflash_inputs_kernel` `:867-1062` kürzt um
  `num_rejected`. Die schnelle Vorberechnung des Kontext-KV vor dem Sampling `dflash2/speculator.py:1053-1082` verlangt gleichförmige
  Länge `num_query_per_req` und **kehrt sonst still in den normalen Pfad zurück**; sie ist ohnehin nur auf (7,0) und bei
  `num_query_per_req == 8` gebaut (`:918-930`) und für die RTX-Stufe 4 nicht aktiv **[abgeleitet]**.
- GDN/KDA-Metadaten: `model_states/mamba_hybrid.py:476-497`: `num_decode_draft_tokens` je Anfrage aus `num_draft_tokens_per_req`;
  `v1/attention/ops/gdn_state.py:424-470` rechnet `spec_query_start_loc`, `spec_token_indx = min(num_spec_decodes*(K+1), query_end)`
  je Runde. Breiten der persistenten Puffer sind K_max-fest (`gdn_attn.py:566-575,591-612,657-662`).
  Capture-Metadaten leiten die Entwurfszahl aus der Abfragelänge ab (`gdn_attn.py:1925-1931`), also stimmt sie auch für Tail-Längen
  **[abgeleitet]**.
- KDA-Schicht `models/glm5next/nvidia/kda.py:628` (fusionierter f_b/g_b-Pfad für 1..8 Token), `:663` `is_recurrent_step = num_tokens <=
  num_spec + 1`, Conv-Update `:893-940` mit `max_query_len = spec_state_indices_tensor.size(-1)` = Breite K_max+1 als **obere Schranke**
  (Länge je Folge aus `query_start_loc`) **[abgeleitet; Test nötig, siehe 6.3]**.
- Sparse-Zielpfad `models/glm5next/sm70/sparse.py:41,163`: `fp8_gemm_max_tokens = min(max_tokens, 8)`, trägt 2..8 Zeilen; K_max ≤ 7.

### 1.4 Entwurf: was kostet der Block

- `get_dflash_model_draft_tokens` (`vllm/config/speculative.py:99-116`) gibt `num_speculative_tokens` zurück, solange kein ngram-assist
  aktiv ist: **Entwurfsblock = k** (heutige Produktion: 4, Abfragezeilen 5). Prüfpunkt `dflash/speculator.py:60-61`
  (`draft_block`, `num_query_per_req = 1 + draft_block`).
- Die gelernte Faltung nutzt `block_size = 1 + draft_block` als **Python-Konstante** im Modell (`model_executor/models/qwen3_dflash2.py:
  116-134,265-271`; Position `% block_size`). Eine Laufzeitänderung des Blocks je Runde ginge dort nicht ohne Umbau.
- Je Entwurfsschritt läuft der Selektor als sequenzieller Ein-Warp-Kernel über `draft_block` Schritte
  (`dflash2/speculator.py:1130-1210`); Kandidatenberechnung `compute_candidates` über `num_reqs*draft_block` Zeilen (`:1438-1452`).
  Der Entwurf besteht aus 5 Schichten, hidden 4096, Zwischengröße 12288, FP8 (`.../GLM-5.3-Flash-DFlash2-Drafter-1.2B-FP8-Block128/config.json`).
  Bei ≤ 8 Zeilen ist das Gewichtslesen bestimmend, die Zeilenzahl kaum **[abgeleitet]**.

### 1.5 Messzahlen und deren Lesart

- Annahme je Position gelesen als **kumulativ** A_j = P(≥ j Entwürfe angenommen). Prüfung gegen die Rundenzeiten (greedy, k=4):
  E = 1 + 0,72 + 0,50 + 0,34 + 0,23 = 2,79 Token je Runde, 2,79 / 55,8 ms = 50,0 tok/s gegen gemessen 50,7. AIfred k=7:
  E = 2,58, 2,58 / 67,9 ms = 38,0 gegen gemessen 36,4 **[abgeleitet; Lesart passt innerhalb etwa 4 %]**.
- Aus den gemessenen Paaren (tok/s × Runde) ergibt sich ein E, das bei k = 3/4/5/7 mit 2,31/2,30/2,39/2,47 um bis zu 0,09 von der
  Positionstabelle abweicht (Tabelle 2,22/2,37/2,47/2,58). Das ist das Rauschen der Messung bzw. die Textgewichtung. Der Regler darf
  sich deshalb nicht auf kleine Unterschiede (< 3 %) stützen (Hysterese, Abschnitt 5).
- Rundenzeit linear: t(n) ≈ t0 + c·n mit c = 4,1…4,2 ms, t0 ≈ 39 ms (k=3: 51,6 gerechnet gegen 51,5; k=7: 68,4 gegen 67,9). **Achtung:**
  diese festen-k-Rundenzeiten enthalten auch den mitwachsenden Entwurfsblock. Unter A (Entwurf immer K_max) ist c nur der
  Prüfanteil und t0 enthält den Entwurf bei K_max; beides ist neu zu messen (Abschnitt 6.4, Schritt 1).

## 2. Frage 1: einfachste richtige Architektur (A)

Der Entwurf liefert jede Runde K_max Token. Der Scheduler plant 1 + n Token. Ablauf mit Stationen:

1. **Scheduler Planung** (`async_scheduler.py:118-162`, Methode `dflash` zulassen, Regler liefert n): Platzhalter der Länge n,
   Tupel-Cache je n statt einem. Keine weitere Änderung; `num_output_placeholders` rechnet mit der geplanten Länge (`:153`),
   `update_from_output` zieht die verworfenen ab (`scheduler.py:2617-2621`) [belegt].
2. **Worker Eingabe**: ohne Änderung (1.3). `num_draft_tokens_per_req = n`.
3. **Zielmodell**: Eingabe 1 + n Zeilen. Gleichförmigkeit `get_uniform_decode_token_count` (`cudagraph_utils.py:198-204`) liefert
   1 + n; Kopie `model_runner.py:1549-1572` verwirft Prefill-Stücke gleicher Form. Ohne Tail-Graph würde `dispatch`
   (`cudagraph_utils.py:498-509`) die Prüfung in den Graphen mit K_max + 1 Zeilen auffüllen; die Auffüllzeilen laufen durch die
   MoE und verteuern die Runde wie echte Zeilen **[abgeleitet; passt zu „wird auf 8 aufgefüllt“ im Gedächtnis zu k-Änderungen]**.
   Deshalb Tail-Graphen (Abschnitt 4).
4. **Rejection**: je Anfrage `cu_num_logits` (1.3). Keine Änderung.
5. **Entwurf nächste Runde**: Eingabe sind die 1 + n Zielzeilen abzüglich verworfener; Ausgabe wieder Breite K_max. Zwischen
   Stufen (`pp_utils.py`) fester Paketbreite K_max.
6. **PP und Rundenversatz**: Die Länge der Runde r+1 steht beim Planen der Runde r fest (Platzhalter). Rückmeldungen der Runde r
   liegen dann noch nicht vor. Mit Warteschlange der Tiefe `pp_size` (`engine/core.py:666-680`, `async_scheduler.py:162`) ist die
   neueste verfügbare Rückmeldung die der Runde r-1, eventuell r-2 **[abgeleitet; im Log nachzählen]**. Der Regler arbeitet damit mit
   1 bis 2 Runden Verzug; bei EMA und Rundenzeiten von 50 ms ist das tragbar. 1Cats Regler hat denselben Verzug.
7. **Erste Runde nach Vorfüllen**: Zustand fehlt, 1Cat nimmt K_max (`async_scheduler.py:111-116`). Besser: Konfiguration
   `initial_tokens` (z. B. 4, die heutige Produktion), weil lange Entwürfe am Anfang bei AIfred-Sampling teuer sind.
8. **Mehrere Anfragen**: gleichförmiges q-len nötig für Graphen; wie bei 1Cat `K_max` (oder feste `initial_tokens`) bei > 1 decodierenden
   Anfragen. In der Produktion ist `max-num-seqs 1`; kein Zweig dafür zu optimieren **[belegt: `--max-num-seqs 1` im llama-swap-Eintrag]**.

Worker-Quellcode bleibt damit unberührt. Alle Änderungen liegen in Konfiguration, Scheduler und der Graphliste.

## 3. Frage 2: Alternative B, Entwurf mit Block 1 + n

- Was B bräuchte: `num_query_per_req`/`draft_block` je Runde (`dflash/speculator.py:60-61,591,725-726`), Entwurfs-Abfragegraphen je Blockgröße
  (`init_cudagraph_manager` `:286-310`, `dispatch_cg_and_sync_dp` mit `uniform_token_count` `:782-790`), Selektor `num_steps=draft_block`
  (konstexpr, Triton-Variante je Wert), `_cached_candidate_*` Breite, Faltung mit eingebackener Blockgröße
  (`qwen3_dflash2.py:271`) als Tensorwert statt Konstante, `torch.compile`-Spezialisierung des Entwurfsmodells je Zeilenzahl,
  `next_spec_steps` an `propose` (nur eagle hat `active_spec_steps`). Die Blockgröße n_{t+1} läge beim Entwurf der Runde t
  ebenfalls schon fest (Scheduler hat sie in `_update_after_schedule` gesetzt, `model_runner.py:1885,1948`), das ginge also.
- Gewinn: Entwurfskosten zwischen Block 1+n und 1+K_max. **[abgeleitet]** klein (Gewichtslesen), aber **[offen]**; die Messung
  in 6.4 Schritt 1 klärt das: forciertes n = 4 mit Entwurf K_max gegen heutige Produktion k = 4. Liegt der Unterschied unter etwa
  1 ms je Runde (< 2 %), bleibt A. Zusätzlicher Vorteil von A: der Entwurf hängt nicht von n ab, also ist eine einzige Spur mit
  forciertem n = K_max genug, um jeden Regler offline zu bewerten (Abschnitt 5.4).
- Empfehlung: **A**.

## 4. Frage 3: Größen und Zustand (was K_max, was n)

| Posten | Größe | Beleg |
|---|---|---|
| Lookahead / Blockzuteilung | K_max + 1 | `vllm/config/vllm.py:683-690` |
| GDN/KDA Spec-Zustandsspalten (`num_speculative_state_tokens + 1`) | K_max | `gdn_attn.py:566-575,591-612`; `kda.py:289-299` (`kda_state_shape(num_spec=…)`) |
| Mamba-„align“-Spec-Blöcke (`num_speculative_blocks`) | K_max | `mamba_attn.py:99-117` |
| Kpool-Tail-Ring | `kpool·next_pow2(cdiv(kpool + K_max, kpool))` = 16 bei K_max 5..8 | `models/glm5next/nvidia/attention.py:178-195` |
| Fused-GDN-Metadaten-Puffer (`placeholder_rows = min(K_max+1, max_bs)`) | K_max | `gdn_attn.py:657-662` |
| `req_states.draft_tokens`, PP-Paket, Sampler-Ausgabe | Breite K_max | `pp_utils.py:97-100`; `rejection_sampler_utils.py:817` |
| Entwurf: Abfrageblock, Kandidaten-Cache, Selektor | K_max | `dflash2/speculator.py:526-566` |
| Zielmodell-Zeilen, Spec-Metadaten je Runde, `cu_num_logits`, `num_accepted_tokens` | **n** | 1.3 |

- Tail-Ring: Er wird mit `pos % ring` beschrieben (`attention.py:166-169`) und ist auf Entwürfe **vor** Annahme ausgelegt; mit
  n ≤ K_max reicht der für K_max bemessene Ring, kleinere n brauchen weniger **[abgeleitet]**. Der Ring bei K_max = 7 hat 16 Plätze
  (Komment/Formel). Die in `~/.claude/.../project_glm_draft_length_k5` genannte offene Frage („Ring 8 bei k=4 ohne Reserve für den Anker“)
  gilt für n > 4 nicht mehr, weil dann 16 Plätze gebaut sind; sie bleibt für K_max = 4 offen.
- Warmup: `vllm/v1/worker/gpu/warmup.py:69` zählt den Tail als 1 Block (Fix 6b1cd5c2c); keine Änderung, aber Test mit K_max = 7 prüfen,
  dass Block-IDs im Pool bleiben (Assert vorhanden).
- KV-Pool: mit K_max = 7 gemessen 550 502 Token, mit k = 4 655 360 (STAND.md, Nachtrag 23.11.), mit k = 5 etwa +12 % gegen k = 7. Die
  Ursachen (Spec-Zustandsspalten, Graph-/Sampler-Puffer auf den pool-begrenzenden Stufen) sind **[abgeleitet, nicht zerlegt]**.
  Entscheidungspunkt für Peuqui: K_max = 7 (volle Wirkung bei Rechnen) gegen K_max = 5/6 (mehr Pool).
- Annahme-Index: Das Zielmodell schreibt Zustandsspalte `num_accepted - 1` als gültigen Stand (`mamba_hybrid.py:644-690`); gilt für
  jedes n ≤ K_max gleich **[abgeleitet; Test 6.3-T5]**.

## 5. Frage 5: Regler (zuerst, weil er die Graphliste bestimmt: n ∈ [min, K_max])

### 5.1 Herleitung der Regel

Je Runde mit n geprüften Positionen: erwartete Token E(n) = 1 + Σ_{j≤n} A_j. Rundenzeit t(n) = t0 + c·n. Durchsatz T(n) = E(n) / t(n).
Position n+1 lohnt genau dann, wenn T(n+1) > T(n), also

    A_{n+1} · t(n)  >  c · E(n)          ⇔          A_{n+1}  >  c · T(n)      (Schwelle θ(n) = c·T(n), dimensionslos mit ρ = c/t0)

Mit ρ = c / t0 und T̃(n) = E(n) / (1 + ρ·n) (Token je Basisrunde): θ(n) = ρ · T̃(n).

### 5.2 Rechenbeispiel mit den gelieferten Zahlen (ρ = 4,2 / 39,0 = 0,108)

AIfred-Sampling, A = 0,62 0,37 0,23 0,15 0,10 0,07 0,04:

| n | E(n) | 1 + ρn | T̃(n) | nächste A | θ(n) = ρ·T̃ | n+1 lohnt? |
|---|---|---|---|---|---|---|
| 1 | 1,62 | 1,108 | 1,462 | 0,37 | 0,158 | ja |
| 2 | 1,99 | 1,216 | 1,637 | 0,23 | 0,177 | ja |
| 3 | 2,22 | 1,324 | 1,677 | 0,15 | 0,181 | **nein** |
| 4 | 2,37 | 1,432 | 1,655 | 0,10 | 0,179 | nein |
| 5 | 2,47 | 1,540 | 1,604 | 0,07 | 0,173 | nein |
| 6 | 2,54 | 1,648 | 1,541 | 0,04 | 0,166 | nein |
| 7 | 2,58 | 1,756 | 1,469 | | | |

→ n* = 3. Gewinn gegen n = 4 nach Modell nur +1,3 % (T̃ 1,677 / 1,655); gemessen wurde k=3 gegen k=4 mit +9 % (44,9 gegen 41,2),
also deutlich mehr als die Tabelle über alle Texte hergibt. Das liegt an der Textsorte (Prosa hat tiefere A_j als der Mittelwert)
und am Rauschen (1.5). Folge: **je Anfrage/Textsorte schätzen, nicht eine globale Tabelle**.

Greedy, A = 0,72 0,50 0,34 0,23 0,17 0,11 0,08: E(n) = 1,72 2,22 2,56 2,79 2,96 3,07 3,15; T̃ = 1,552 1,826 1,934 1,948 1,922
1,863 1,794 (n = 1..7); θ(3) = 0,209 gegen A_4 = 0,23 (ja, knapp), θ(4) = 0,210 gegen A_5 = 0,17 (nein) → n* = 4, Plateau n = 3..5
(T̃ 1,934 / 1,948 / 1,922: 1 % Unterschied) = „Code flach 3..5“. Rechnen mit hoher Annahme (nur **Beispiel**, nicht gemessen: A_j = 0,85^j):
T̃ steigt bis n = 7 (2,40 / 2,58 / 2,69 / 2,74 / 2,76 für n = 3..7) → n* = 7, passend zu „Rechnen greedy k = 5..7“.

### 5.3 Schätzer (verallgemeinerter 1Cat-Regler, methodenunabhängig)

- Zustand je Anfrage: je Position j = 1..K_max Zähler (s_j, t_j), exponentiell gewichtet (alpha), Anfangswert aus Konfiguration
  (`prior_conditional_acceptance`, z. B. 0,7 mit Gewicht ≈ 2 Versuche). Beobachtung je Runde mit n Entwürfen, a angenommen:
  Position j ≤ n wird **versucht**, wenn a ≥ j-1 (t_j += 1), und **gelingt**, wenn a ≥ j (s_j += 1). Das ist die bedingte Annahme
  c_j = s_j/t_j; A_j = Π_{i≤j} c_i. (1Cat nimmt eine feste geometrische Form A_j = p^j; unsere Messung zeigt Verhältnisse
  0,69/0,68/0,68/0,74/0,65/0,73, also grob geometrisch im Mittel. Die Einzelschätzung je Position braucht keine Annahme darüber,
  kostet sieben Zähler.)
- Zensur: Positionen > n werden nicht beobachtet. Für sie wird c_j := c_n (letzte beobachtete bedingte Annahme) fortgeschrieben,
  für die Frage „lohnt n+1“ genügt A_{n+1} = A_n · c_n. Dazu wie bei 1Cat „sofort +1 bei voller Annahme (a = n)“ (`async_scheduler.py:78-80`),
  damit der Regler nach oben nicht hängen bleibt.
- Entscheidung: n* = argmax_{n ∈ [min, K_max]} T̃(n) (höchstens sieben Terme, keine Bänder). Wechsel nur, wenn T̃(n*) > (1 + margin)·T̃(n_alt)
  (Hysterese gegen Rauschen, margin ≈ 0,02 wegen 1.5). Alle Graphen für n sind erfasst, Sprünge um mehr als 1 sind zulässig.
- Gilt n = 1 als Untergrenze? Ja (`min_tokens` 1); n = 0 würde ohne Entwurf laufen, dort ist die Annahme in dieser Verteilung nie
  so niedrig (A_1 = 0,62 ≫ θ ≈ 0,18).

### 5.4 Konfiguration (KernelConfig, keine Umgebungsvariablen)

Neues Feld `kernel_config.adaptive_verify` (Dataclass `AdaptiveVerifyConfig` neben `Sm70DFlash2Config`-Muster; Aufruf
`--kernel-config '{"adaptive_verify":{…}}'`). Muss in `KernelConfig.compute_hash` unter `ignored_factors` (`vllm/config/kernel.py:1486-1505`)
eingetragen werden, sonst ändert sich der Compile-Hash und alle Caches rechnen neu (reines Scheduler-Feld, **[belegt]** Muster der
Liste).
Felder: `enabled` (off/on, kein `auto`), `min_tokens` (1), `max_tokens` (Standard `num_speculative_tokens`), `initial_tokens`,
`ema_alpha` (0,28 wie 1Cat), `prior_conditional_acceptance`, `prior_weight`, `verify_cost_ratio` (ρ), `switch_margin`.
Erweiterung für 1Cat-Regler: die Methode `mtp` behält ihre festen Bänder unverändert (Merge-Treue zu 1Cat main, Datei ist von 1Cat,
`async_scheduler.py`); Entscheidungsfunktion wird je Methode gewählt (eine Stelle). Eine spätere Vereinheitlichung (Argmax auch für MTP)
nur nach A/B.

### 5.5 Kalibrierung (ohne GPU-Zeit für den Regler selbst)

1. **Kosten:** forcierte Läufe n = 1..K_max (Regler `min_tokens = max_tokens = n`) bei Entwurf K_max: t(n) kurz, bei 150k und Maximum
   → t0, c (und ob c mit dem Kontext wächst; falls ja ρ je Kontextklasse oder Online-Schätzung). Ergebnis ersetzt 4,2/39.
2. **Annahme:** ein Lauf mit forciertem n = K_max liefert unzensierte Annahmefolgen je Runde und Textsorte (ein Entwurf hängt in A
   nicht von n ab). Daraus A_j je Textsorte (Prosa, Code, Rechnen, AIfred-Sampling und greedy).
3. **Regler offline** auf diesen Folgen simulieren (Python, CPU): alpha, prior, margin, initial wählen; Zielgröße Σ Token / Σ t(n).
   Näherung: Verteilungsverschiebung durch geänderte Pfade ist klein [abgeleitet].

## 6. Frage 4: CUDA-Graphen

### 6.1 Welche
- **Zielmodell Prüfrunde**, jede PP-Stufe (alle fünf Worker laufen `ModelCudaGraphManager`): FULL-Graphen für Abfragelängen 1 + n,
  n = min_tokens..K_max, nur Einzelanfrage (`num_reqs = 1`). Für K_max = 7: Längen 2..8 (heute Produktion k=4: nur 5).
- GLM-Eintrag: `cudagraph_capture_sizes` weiter `[K_max+1]`; die Tail-Längen kommen über `decode_query_lens` (siehe 6.2) in
  `_capture_sizes` (Pfad „Legacy“, `cudagraph_utils.py:305-318`, da GLM Methode `dflash` ist und `_use_split_sm70_mtp_cudagraphs` nur für `mtp`
  gilt, `:117-125`). **[belegt]**
- **Entwurf:** nichts (A).
- Sampler-Graphen (`dflash2/sampler_graph.py`): nicht aktiv auf RTX/TP1.
- Kontext-KV-Graphen des Entwurfs (`dflash2/speculator.py:918-995`): nicht aktiv (Stufe 4 ist RTX).

### 6.2 Wiederverwendung statt Duplikat (eine Dispatch-Stelle)
Heute gibt es **zwei** Einträge mit demselben Zweck in `CudaGraphManager.__init__`:
`_sm70_dflash2_tail_graphs` (`cudagraph_utils.py:236-253`, nur `decode_query_len == 8`, nur V100/Turing-Prüfung `_worker_device_is_pre_ampere`,
Flag `tail_cudagraphs`, Längen 7..1, koppelt zusätzlich die Attention-Tail-Route in `:595-615`) und `_adaptive_mtp_tail_graphs` (`:256-266`, Längen
`decode_query_len-1 … 3`). Beide verwenden dieselbe Einschränkung `:340-352` (nur Einzelanfrage-Tails).
Vorschlag: eine Hilfsfunktion `tail_query_lens(vllm_config, decode_query_len) -> tuple[int, ...]` (Quelle: `AdaptiveVerifyConfig` oder
die bestehenden Flags) und ein Attribut `self.tail_query_lens`; `decode_query_lens`, die Einschränkung `:340-352` und die Legacy-Erweiterung
`:315-318` lesen nur noch dieses Attribut. Das Attention-Verhalten (`:595-615`) bleibt am Flag `tail_cudagraphs` (anderer Zweck).
Die Gleichung `decode_query_len == 8` fällt weg (Verallgemeinerung auf K_max).
**[offen]:** `_qsa_short_graphs` (`:630-648`) und `_compressed_context_graphs` (`:660-678`) variieren nur Graphen mit voller Länge
`decode_query_len`; falls für GLM `qsa_dense_short_context` oder `compressed_buckets` aktiv sind, fehlen sie für Tails. Boot-Log prüfen
(`QSA short-context graph variants captured …`).

### 6.3 Speicher und Zeit
Je Stufe sechs weitere Graphen (K_max = 7). Private Graphspeicher je Graph ist gegenüber dem gemeinsamen Pool klein, auf den
pool-begrenzenden Stufen (S0, S4) aber nicht null **[abgeleitet]**. **[offen]** Messung: Capture-Dauer und GiB je Stufe aus dem Boot-Log
(„Graph capturing finished“), GPU-Spitzen in der A/B (`gpu_log.sh`). Falls zu viel: `min_tokens` = 2 und Obergrenze `max_tokens` = 6.

## 7. Frage 6: Änderungsliste, Tests, Abnahme

### 7.1 Dateien
1. `vllm/config/kernel.py`: neue Dataclass `AdaptiveVerifyConfig`, Feld `adaptive_verify` in `KernelConfig` (`:755`), `ignored_factors`
   (`:1486`), Auflösung `resolve_adaptive_verify(speculative_config, kernel_config, num_spec_tokens)`, ersetzt `is_adaptive_mtp_enabled`
   (`:1727`) und `adaptive_mtp_min_k` (`:1761`) als einzige Quelle für Scheduler und Graphmanager (SSOT). `mtp` erhält sein Verhalten
   über den bestehenden Pfad der Richtlinie.
2. `vllm/v1/core/sched/async_scheduler.py`: `__init__` (`:33-45`), Regler (`:47-100`) als Funktion je Methode, Zustand je Position,
   `_select_adaptive_batch_k` (`:102-116`), Tupel-Cache je n (`:137-140`), Protokollzeile beim Wechsel (INFO, begrenzt).
3. `vllm/v1/worker/gpu/cudagraph_utils.py`: `tail_query_lens` (`:236-296`), Einschränkung (`:340-352`), Legacy-Erweiterung (`:315-318`).
4. `vllm/v1/core/sched/scheduler.py`: keine Änderung nötig (Haken `:2629` liefert schon Entwurfs-/Annahmezahl).
5. Worker/Entwurf/Rejection/PP: **keine Änderung** (A). Kontrolle: `model_runner.py:263,341` bleiben `1 + K_max`.
6. Dokumentation: STAND.md Nachtrag, llama-swap-Eintrag (nur nach Abnahme) mit `num_speculative_tokens` K_max,
   `cudagraph_capture_sizes` `[K_max+1]`, `--kernel-config`-Zusatz.
7. Optional: Metrik/Log mit Histogramm von n und je Position A_j aus `SpecDecodingStats` (`scheduler.py:3385-3395`, Zähler existieren).

### 7.2 Tests (je einer muss am alten Code fehlschlagen)
- **T1 Scheduler, Planlänge** (`tests/v1/core/test_async_scheduler.py`-Muster, `create_scheduler(num_speculative_tokens=7, async_scheduling=True, …)`
  mit Methode `dflash`; `create_scheduler` hat heute keinen Methodenparameter `tests/v1/core/utils.py:99`, also erweitern): Folge von Rückmeldungen
  mit voller/niedriger Annahme → die geplanten Entwurfsplatzhalter der nächsten Runde haben Länge n gemäß Regel. Alt: immer K_max (Regler nur für `mtp`).
- **T2 Reglerfunktion rein** (Tabellentest mit den Zahlen aus 5.2): AIfred-A → 3, greedy-A → 4, A = 0,85^j → 7. Alt: Funktion fehlt.
- **T3 Schätzer-Zensur:** Beobachtungen mit n = 3 verändern c_4.. nicht; volle Annahme hebt n um 1. Alt: fehlt.
- **T4 Graphliste:** `tests/v1/worker/test_sm70_long_attention_graphs.py::test_tail_capture_and_dispatch_from_real_initialization`
  (Zeile ~136-200) erweitern: Methode `dflash`, `decode_query_len` 5 und 8, kein `tail_cudagraphs`-Flag → `decode_query_lens` = 8..2 und nur
  Einzelanfrage-Tails; Dispatch von 3 Token liefert Graph mit genau 3. Alt: Tail nur bei `== 8` und gesetztem Flag.
- **T5 Zustandsrichtigkeit (GPU, kleines Modell oder Mini-Prüfpunkt)**: Greedy-Texte gleich bei forciertem n = 1..K_max und gleich dem heutigen k = n.
  Das fängt Tail-Ring, KDA-Conv und Zustandsspalte ab. Alt: n > fest nicht einstellbar (Test-Fixture mit festem n).
- **T6 PP/Async CPU** (`tests/v1/core/test_pp_spec_decode_inflight.py`-Muster): `pipeline_parallel_size=2`, wechselnde n in Folge, Prüfung
  `num_output_placeholders` ≥ 0 und `num_computed_tokens` konsistent. Alt: n konstant.
- **T7 Hash:** zwei `KernelConfig` mit/ohne `adaptive_verify` haben denselben `compute_hash` (verhindert Cache-Neuberechnung).
- **T8 Konfiguration:** `min_tokens > max_tokens`, `max_tokens > K_max`, unbekannte Methode → ValueError (kein stilles Zurückfallen).

### 7.3 Messungen (Prüfstufe 2, Entscheidung nach Regeln in CLAUDE.md)
1. Kosten: forciertes n gegen heutige Produktion k = n (n = 3, 4, 5), AIfred-Sampling; entscheidet A gegen B (Abschnitt 3) und ρ.
2. Abnahme A/B T P T P, P = Produktion k=4 (v14): AIfred-Sampling (alle Textsorten, Tabelle nach Textsorte), greedy, GSM8K-30 (greedy Decode,
   Ziel: k=4-Verlust −5,7 % wird ≥ 0), Decode kurz/lang nach 8,7k, Langkontext 150k/250k/Maximum in allen Armen (Streuung der Arme
   34–54 tok/s beachten, ≥ 2 Arme), KV-Pool je K_max, GPU-Spitzen je Karte (`gpu_log.sh`), Boot-Dauer und Capture-GiB.
3. Texte: Qualitätsfragen (Kuanda: Zurückweisen = bestanden) händisch lesen; Wortgleichheit zu k = 4 nicht zu erwarten bei Sampling.
4. Regler-Betrieb: Histogramm n je Textsorte aus dem Log; Verzug (Runden zwischen Beobachtung und Wirkung) nachzählen.
5. Erfolgskriterium (Stufe 2, modellspezifisch ≥ 3 %): AIfred-Sampling gegen k=4 ≥ +3 % bei Prosa und Rechnen nicht schlechter, Langkontext nicht schlechter, Pool ≥ Wunsch von Peuqui.

## 8. Risiken und offene Punkte (alle noch zu belegen)

- ρ und c unter Architektur A (Entwurf K_max) nicht bekannt; Kontextabhängigkeit von c bei langem Kontext unbekannt (Abschnitt 5.5).
- Tail-Graphen auf GLM: Verhalten von KDA-Conv/Zustandsspalte, fusionierten GDN-Metadaten (`mamba_hybrid.py:517-545`) und Tail-Ring
  bei n < K_max nur abgeleitet; Test T5 ist Pflicht vor dem Boot in Produktion. Zusätzlich die Vorab-Quellprüfung aus CLAUDE.md
  (`scan_arch_gates.sh`) für die Kerne bei 2..8 Zeilen auf V100 UND RTX.
- Pool −16 % bei K_max = 7 gegen k = 4 (Abschnitt 4).
- Regler-Verzug 1–2 Runden (PP5), Schätzung einer einzigen Anfrage über wenige Runden; Hysterese und Prior entscheiden die ersten Runden
  (offline simulieren).
- 1Cat-Merge: `async_scheduler.py` und `kernel.py` sind 1Cat-Dateien; minimale, klar abgegrenzte Änderung (Entscheidungsfunktion je Methode) hält den
  nächsten Merge billig. Als PR an 1Cat nur nach eigener Abnahme.
- Nicht belegt: ob `tail_cudagraphs` standardmäßig an ist (`vllm/config/flash_v100.py:501` Standard `None`, Auflösung nicht verfolgt) und ob die heutige
  Produktion mit k = 4 Tail-Graphen für DFlash2 bereits erfasst (Bedingung `decode_query_len == 8` ist bei k = 4 falsch, also nein **[belegt]**).
