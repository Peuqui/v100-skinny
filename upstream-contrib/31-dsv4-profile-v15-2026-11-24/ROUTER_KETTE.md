<!-- 24.11.2026: Zuordnung durch einen Sonnet-Agenten (nur gelesen, Baum v16). Abgeleitete Zuordnungen sind als solche markiert; Profil-Bestätigung steht aus. -->

# Bericht: MoE-Router-Kette DeepSeek-V4-Flash im Decode (nur gelesen, nichts geändert)

Baum: `/home/mp/Projekte/vllm-research/1Cat-vLLM-v16` (Zweig prod-v16). Alle Pfade unten relativ dazu.

Hinweis vorab: Der Produktionseintrag in `/home/mp/.config/llama-swap/config.yaml` setzt `PYTHONPATH=…/1Cat-vLLM-v15`, nicht v16. Ich habe nur v16 gelesen. Ob v15 an den zitierten Stellen gleich ist, habe ich nicht geprüft.

Relevante Eintragsdaten: `--dtype half`, PP5, `--max-num-seqs 1`, DSpark mit `num_speculative_tokens 5`, `cudagraph_capture_sizes [5,6]`, `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`, `VLLM_PP_LAYER_PARTITION=10,8,8,8,9`. Es gibt kein `moe_backend` (also "auto"), kein `VLLM_SM70_DSV4_FP13_GEMV` (Default True, `vllm/envs.py:399`) und kein `sm70_skinny_moe` (Default True, `vllm/config/kernel.py:1031`).

Modellparameter (aus `/home/mp/models/DeepSeek-V4-Flash-284B-A13B-MXFP4-FP8-DSpark/config.json`): hidden 4096, 256 geroutete Experten, top-k 6, 1 Shared-Experte, moe_intermediate 2048, `routed_scaling_factor` 1,5, `norm_topk_prob` True, `scoring_func` sqrtsoftplus, `topk_method` noaux_tc, `num_hash_layers` 3, n_group/topk_group None, swiglu_limit 10.

## 0. Auflösung des scheinbaren Widerspruchs (FP13 gegen cuBLAS-Kernel)

Beide Aussagen stimmen. Der FP13-Pfad ist nur für genau eine Zeile gebaut. Im DSpark-Decode hat das Gate aber 5 oder 6 Zeilen und fällt deshalb auf cuBLAS.

- `GateLinear.forward` ruft zuerst `maybe_sm70_dsv4_fp16_gemv` (`vllm/model_executor/layers/fused_moe/router/gate_linear.py:99-113`).
- Der Vertrag `_has_sm70_dsv4_gemv_contract` verlangt `x.shape == (1, 4096)` (`vllm/models/deepseek_v4/sm70/gemv.py:427`). Er verlangt außerdem FP16 für x und Gewicht (`:423-424`) und `is_device_capability((7,0))` (`:419`). Bei 5 oder 6 Zeilen liefert `can_use_sm70_dsv4_fp13_gemv` (`:446-461`) False, die Funktion gibt `None` zurück (`:488-489`).
- Das Packen der FP13-Gewichte passiert beim Laden (`vllm/model_executor/layers/linear.py:367-374` → `gemv.py:267-297`). Es hängt nicht von der Zeilenzahl ab, nur von der V100-Kennung (`gemv.py:277`). Die gepackten Puffer liegen also auf den V100-Stufen im VRAM, werden aber im 5/6-Zeilen-Decode nie gelesen. Das ist meine Ableitung aus dem Vertrag. Ein Boot-Log oder Trace dazu habe ich nicht gesehen. Gebraucht würde der Puffer nur bei Schritten mit genau einem Token (z. B. eager, nicht im Graphen). Ob es solche Schritte in der Produktion gibt, ist nicht geprüft.
- Auf RTX (SM75) wird der FP13-Pfad nie gepackt und nie genommen (`gemv.py:277` und `:419`, beide verlangen (7,0)).
- Folge: Das Gate läuft auf V100 und RTX identisch über Tier 3, also cuBLAS (siehe Frage 1). Der Satz in ANALYSE.md Zeile 232 "Gate bleibt BF16-Skinny" stimmt nicht. Das Gate liegt wegen `--dtype half` im VRAM als FP16 (aus BF16 im Checkpoint umgewandelt).

## 1. Gate-Pfad im Decode

Aufrufkette (V100 und RTX gleich):

1. `DeepseekV4MoE.forward` → `_forward_fused_moe` (`vllm/models/deepseek_v4/nvidia/model.py:552-558`, `592-610`). Das Gate ist als `gate=self.gate` im `FusedMoE` registriert (`:533-535`, `is_internal_router`). Deshalb übergibt das Modell nur Platzhalter (`router_logits=hidden_states`, `:596-602`) und rechnet das Gate nicht doppelt. GLM hatte dieses Doppelt-Rechnen und wurde in Commit 0ef69f4e6 korrigiert, DeepSeek hatte es nie.
2. `MoERunner._forward_impl` ruft `self.gate(hidden_states)` (`vllm/model_executor/layers/fused_moe/runner/moe_runner.py:993-1010`). Die Funktion läuft innerhalb des Custom-Ops `moe_forward_shared` (`:249-263`, undurchsichtig für Inductor). `maybe_tp_sharded_router_topk` greift nur bei mindestens 1024 Zeilen und Prefill-Sequenzparallelität, also nicht im Decode.
3. `GateLinear.forward` (`gate_linear.py:94-135`) hat diese Stufen:
   - FP13/FP16-GEMV (`:99-113`): nur bei genau 1 Zeile, siehe Abschnitt 0.
   - Tier 1 `dsv3_router_gemm` (`:116`): braucht Hopper/Blackwell, `allow_dsv3_router_gemm` ist False (`:39-44`, `:64-68`).
   - Tier 2 `torch.mm(..., out_dtype=float32)` (`:125`): ebenfalls nur Hopper/Blackwell.
   - Tier 3 (`:129-135`) ist der echte Pfad. x ist schon FP16, also kein Cast des Eingangs. `ReplicatedLinear.forward` → `UnquantizedLinearMethod._apply_impl` (`vllm/model_executor/layers/linear.py:420-472`). Die Zweige davor greifen nicht: DFlash2-Marken fehlen, Online-QPN8 ist nur Qwen4 und Default aus, der Dense-F16-Pfad ist Default aus (`VLLM_SM70_ENABLE_DENSE_F16_FASTPATH`, `envs.py:~6070`). Also endet es bei `dispatch_unquantized_gemm()` → `torch.nn.functional.linear` (`vllm/model_executor/layers/utils.py:122-128`, `:351-357`). Das ist ein cuBLAS-FP16-GEMM [6,4096] × [4096,256], Ausgabe FP16.
   - Danach `output.to(torch.float32)` (`gate_linear.py:133-134`), weil `out_dtype=float32` (`model.py:443-449`).
4. Die Präzisionsrichtlinie (`vllm/utils/torch_utils.py:50-72`) schaltet reduzierte FP16-Reduktion und FP16-Akkumulation ab, also FP32-Akkumulation. Split-K bleibt je nach Voreinstellung erlaubt.

Zuordnung der Profilkerne:

- `Kernel2` + `splitKreduce`: cuBLAS-Split-K-GEMM des Gates (Hauptkern und Reduktion). Das ist abgeleitet aus Aufrufkette und Namen. Die Kernnamen selbst habe ich nicht aus einem Trace gesehen.
- `unrolled` (8,4 µs): vermutlich der PyTorch-Elementweise-Kern für den Cast FP16 → FP32 (`gate_linear.py:134`). Das ist nicht belegt. Die Reihenfolge im Layer-Trace und die 1Cat-Notiz "plus reducer/cast" (`docs/design/sm70_deepseek_v4_fp16_gemv.md:10`) sprechen dafür. 8,4 µs für 1536 Elemente ist aber auffällig lang.
- Achtung: Die Rollenzuordnung in `analyse_nsys_dsv4.py:169-170` ordnet `Kernel2`/`splitKreduce`/`gemmSN_TN` nach Position (FFN-Seite) dem Router zu. Dieselben Namen gehören auf der Attention-Seite zu den Compressor-GEMMs, und `Kernel2 (8,1010,1)` ist der LM-Kopf. Die Zuordnung zum Gate ist also eine Positionsableitung, keine Namensableitung.

Umschaltbedingungen:

- Zeilenzahl 1 gegen mehr (nur FP13/FP16-GEMV, nur V100).
- Env `VLLM_SM70_DSV4_FP13_GEMV` (Default 1) und `VLLM_SM70_DSV4_FP16_GEMV` (Default 0).
- Hash-Schichten 0–2: `tid2eid` ist am Gate-Objekt gesetzt (`model.py:453-466`). Das ändert die Gate-Berechnung nicht, nur den Top-k-Kern (`USE_HASH`). Hash-Schichten sind nur auf Stufe 0 (Partition 10,8,8,8,9, also RTX), ANALYSE.md nennt sie "Schichten 0–1", der Code `< num_hash_layers` mit Wert 3 meint 0–2.
- Unbelegt: das Gate der DSpark-Entwurfsschichten auf Stufe 4 (`vllm/models/deepseek_v4/nvidia/dspark.py`) habe ich nicht verfolgt.

## 2. Top-k und Gruppenrouting

Aufrufkette: `FusedTopKBiasRouter._compute_routing` (`vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py:331-353`; Auswahl in `router_factory.py:187-198`, weil Bias oder Hash-Tabelle vorhanden) → `fused_topk_bias` (`:160-229`) → `vllm_topk_softplus_sqrt` (`:106-144`) → `ops.topk_hash_softplus_sqrt` (`vllm/_custom_ops.py:2607-2628`) → `torch.ops._moe_C.topk_softplus_sqrt` (`csrc/moe/torch_bindings.cpp:26-30`) → `topkGatingSoftplusSqrt` in `csrc/moe/topk_softplus_sqrt_kernels.cu`. Zuweisung: `topk_weights` FP32 [M,6], `topk_ids` int32 [M,6], `token_expert_indices` int32 [M,6] wird angelegt, aber danach nicht gebraucht (`fused_topk_bias_router.py:188-190`).

Was der Kern rechnet (`topk_softplus_sqrt_kernels.cu`):

- Eingang FP32-Logits [M,256] (auch FP16 und BF16 möglich, `:705-722`). Dispatch für 256 Experten: `:591-593`, 4 Warps je Block, 16 Byte je Laden. Ein Warp je Zeile (`THREADS_PER_ROW` 32, `VPT` 8), also bei 6 Zeilen 2 Blöcke zu je 128 Threads. Mit PDL-Attribut gestartet (`:481-491`, auf SM70/75 ohne Wirkung, die Anweisungen `griddepcontrol` gibt es nur ab SM90).
- Score: softplus mit `val > 20 ? val : __logf(1 + __expf(val))`, danach `sqrtf` (`:233-309`). Das sind schnelle Intrinsics, nicht `log1p`/`exp` der Standardbibliothek.
- Nicht-Hash-Schichten (`:301-421`): Auswahlscore = Score + `e_score_correction_bias` (FP32 [256], `:310-317`). Top-k als sequenzielles Argmax (k = 6 Durchläufe, je 5 Shuffle-Schritte für 32 Threads je Zeile), Gleichstand gewinnt der niedrigere Index (`:361-364`). Gewählter Eintrag wird mit −10000 gelöscht. Gewicht = `max_val − bias` (`:379-381`). Das ist (Score+Bias)−Bias in FP32 und nicht bitgleich zum Score. Normierung: `scale = routed_scaling_factor / Summe` (Summe nur von Lane 0, in Auswahlreihenfolge addiert, `:410-420`), mit `renormalize` = True und Faktor 1,5.
- Hash-Schichten 0–2 (`USE_HASH`, `:237-300`): Indizes aus `tid2eid[input_ids]` (int32 [129280,6], Eingabe-IDs ebenfalls int32, nicht per Modell geprüft, `model.py:1119` wandelt nur bei MegaMoE in int64), Gewichte aus dem unverzerrten Score, kein Bias, gleiche Normierung × 1,5.
- Ausgabe `routed_scaling_factor` ist bereits im Gewicht (der Runner skaliert nicht noch einmal, weil `apply_routed_scale_to_output` False ist, `vllm/model_executor/layers/fused_moe/layer.py:277-283`).
- Keine Gruppen (n_group/topk_group None).

Gruppenrouting danach: `group_routing(topk_ids)` (`vllm/model_executor/layers/fused_moe/sm70/routing.py:82-106`), aufgerufen in `Mxfp4SkinnySm70Experts._apply_grouped` (`vllm/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py:835`). Das ist der Triton-Kern `_group_routing_kernel` (`routing.py:47-79`), ein Programm mit einer Warp, bitonische Sortierung von Schlüsseln `expert*BLOCK+slot` (`:64-66`), bis 128 Slots (`:17`), bei 6×6 = 36 Slots Block 64. Ausgaben alle int32: `perm` [S], `gids` [S], `goff` [S+1] (S = Zeilen × 6). Kein Kartengate, läuft auf V100 und RTX gleich.

Was der Skinny-Up-Kern braucht:

- Für W13 und W2: `perm`, `gids`, `goff`, `top_k` (`skinny_sm70_moe.py:891-943`, Op `torch.ops._C.skinny_moe_qpn_sm70`). `topk_ids` selbst geht nur durch `group_routing`.
- `topk_weights` wird erst am Ende gebraucht, im Triton-Kern `_mtp_weighted_reduce` (`vllm/model_executor/layers/fused_moe/sm70/reduction.py:54-98`, aufgerufen `skinny_sm70_moe.py:889`).
- Der Triton-Ersatz `skinny_moe_e8m0_m8n8k4_triton` greift bei über 13 Zeilen je Experte und nur für Fragmentreihenfolge M8N8K4 (V100), bei 36 Slots auf 256 Experten also nicht (`:915-923`).
- Dass `Mxfp4SkinnySm70Experts` in der Produktion tatsächlich gewählt wird, ist aus dem Code abgeleitet (`vllm/model_executor/layers/quantization/mxfp4.py:846-875` wählt es bei `moe_backend == "auto"` und `skinny_backend_admitted`, Bedingungen u. a. FP16-Aktivierungen, kein EP, SM70 oder SM75, `skinny_sm70_moe.py:667-724`). Aus dem Boot-Log (`moe_kernel_selections`) habe ich es nicht bestätigt.

## 3. Shared-Expert zwischen Gate und Top-k

Nur Codereihenfolge, keine Datenabhängigkeit.

- Der Shared-Experte bekommt dieselbe Eingabe `hidden_states` wie das Gate (`moe_runner.py:757-770`) und hängt weder vom Gate noch vom Top-k ab. Seine Ausgabe wird erst am Ende addiert (`:882-883`, `:941-959`).
- Reihenfolge in der Produktion: Gate in `_forward_impl` (`:993-1010`), dann `_apply_quant_method`, dessen erste Anweisung die Shared-Experten im Modus NO_OVERLAP ausführt (`:625-627`), dann `router.select_experts` (Top-k, `:644-650`), dann der Experten-Kern. NO_OVERLAP gilt, weil `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` den Zusatzstrom abschaltet (`runner/shared_experts.py:72-84`, `:110-130`) und der Skinny-Kern nicht intern überlappt (`modular_kernel.py:1721-1725`, abgeleitet).
- Eine Fusion Gate+Top-k braucht also die Shared-Rechnung nicht zu berücksichtigen, außer man will sie später in der Reihenfolge verschieben. Ohne die Env wäre die Reihenfolge anders (Aux-Stream parallel), die Entscheidung "Shared-Stream bleibt aus" steht in der Memory.

## 4. Vorhandene fusionierte Router-Bausteine

Hinweis: Die GLM-Commits sind keine Fusion. "GLM-Router einmal je MoE-Schicht" (0ef69f4e6) beseitigt ein doppeltes Gate-GEMM (Test `tests/models/glm5next/test_moe_router_once.py`). "Routing-Gruppenkern als Sortierung" (ca2c1b65a, davor 9712e0edf) ist der `group_routing`-Kern oben und wird von GLM und DeepSeek geteilt (beide nutzen Skinny-Experten).

Wiederverwendbar oder als Vorlage:

1. `group_routing` (`sm70/routing.py`): schon geteilt. Eine Fusion hinter dem Top-k kann das Sortieren der 36 Schlüssel im selben Kern übernehmen. Die Ausgabe-Konvention (stabil nach Experte, dann Slot) ist festgelegt durch `group_routing_reference` (`routing.py:23-44`).
2. `benchmarks/benchmark_sm70_deepseek_v4_fused_router.py`: Last-CTA-Prototyp (1Cat) aus FP16-GEMV, sqrt-softplus, Bias, Top-6 und Normierung. Er ist nur für eine Zeile (M=1), nur Nicht-Hash, hat `ROUND_TO_FP16` für die Rundung wie bei cuBLAS und nutzt `tl.log`/`tl.exp`. Laut `docs/design/sm70_deepseek_v4_fp16_gemv.md:95-117` von 1Cat verworfen: 40 Schichten TP8 M=1, −0,065 ms (128 CTAs) und −0,134 ms (256 CTAs) gegen 0,5201 ms, unter deren Schwelle von 0,2 ms je Token. Unsere Policy hat keine Mindestschwelle mehr. Die Messung ist M=1 auf TP8, nicht unser M=6. Der Einzelschicht-Test überschätzte wegen L2-warmem Gate-Gewicht (2 MiB).
3. `vllm/model_executor/kernels/mhc/deepseek_v41_router.py` (`route_topk`, Triton, ein Programm je Token, Eingang FP32-Logits und Bias, ids int64): nur vom Modell DeepSeek-V4.1 benutzt (`vllm/models/deepseek_v41/model.py:40`, `:341`), nicht von V4. Unterschiede: `total + 1e-20` statt `selected_sum > 0 ? : 1`, kein Hash, int64-IDs, Gleichstand per `min(index)`. Als Vorlage für einen Triton-Epilog geeignet, nicht bitgleich zum C++-Kern.
4. Qwen-eigene Router-Kerne für SM70: `qwen38_router_batch_sm70_out` (Gate-GEMM für 2 bis 64 Zeilen, nur FP16 [512,2560], `csrc/sm70_turbomind/ops/qwen38_gdn_input_batch_sm70.cu:1371-1395`) und `vllm_sm70_router.top10`/`select_quantize` (`csrc/sm70_turbomind/ops/sm70_router.cu`, nur SM70, 512 Experten, Hidden 2560). Formen fest verdrahtet. Als Vorbild für ein Mehrzeilen-Gate-GEMM nützlich, nicht direkt wiederverwendbar.
5. GLM-Top-k: `grouped_topk_fused_small_expert_count_kernel` (`csrc/moe/grouped_topk_kernels.cu:672-884`, Aufruf `:887-972`, Python `fused_grouped_topk` `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:29-69`, Auswahl bei GLM wegen `n_group` 1 und Bias über `GroupedTopKRouter`, `router_factory.py:142-167`). Scoring nur `SCORING_NONE`/`SCORING_SIGMOID` (`:441-443`), kein Hash. Für DeepSeek bräuchte man einen neuen Scoring-Zweig sqrt-softplus. Die Auswahl- und Gewichtssemantik (Score+Bias für die Wahl, unverzerrter Score als Gewicht, Normierung, Skalierung) ist aber dieselbe.

Unterschiede GLM gegen DeepSeek (Config `/home/mp/models/GLM-5.3-Flash-NVFP4-MTP/config.json`):

| Eigenschaft | GLM-5.3 | DeepSeek-V4 |
|---|---|---|
| Experten / top-k | 288 / 8 | 256 / 6 |
| Scoring | sigmoid | sqrt(softplus) |
| Gruppen | n_group 1, topk_group 1 | keine |
| Skalierung | 2,5 | 1,5 |
| Renormalisieren | ja | ja |
| Bias | ja | ja (Hash-Schichten 0–2: Tabelle statt Bias) |
| Logit-Typ | FP32 (`moe_router_dtype`) | FP32 |
| Router-Klasse | `GroupedTopKRouter` → `ops.grouped_topk` | `FusedTopKBiasRouter` → `topkGatingSoftplusSqrt` |
| Experten-Kern | Skinny-NVFP4 (gleicher `group_routing`) | Skinny-MXFP4 (gleicher `group_routing`) |

## 5. Tests und Benchmarks

- `tests/kernels/moe/test_topk_softplus_sqrt.py`: Torch-Referenz, Nicht-Hash und Hash (`test_fused_topk_softplus_sqrt`, `test_fused_topk_softplus_sqrt_hash`), Zeilen 1/33/128, Experten 128–512, top-k 6/8/16, Faktor 1,0/1,5/2,5. IDs exakt, Gewichte nur `atol=2e-2, rtol=1e-2`, beide sortiert verglichen. Für eine bitgenaue Fusion zu locker, und die echte Zeilenzahl 6 fehlt.
- `tests/kernels/moe/test_fused_topk.py`, `test_routing.py`, `test_grouped_topk.py` (GLM-Weg), `tests/models/test_deepseek_v41_contract.py`.
- `tests/kernels/moe/test_sm70_group_routing.py` (GPU) und `test_sm70_group_routing_cpu.py` (Triton-Interpreter): Gruppenrouting gegen `group_routing_reference`.
- `tests/kernels/test_sm70_deepseek_v4_fp16_gemv.py` (FP13/FP16-GEMV im Graphen, nur V100), `tests/kernels/test_sm70_fused_fp16_aux.py`.
- `tests/kernels/moe/test_skinny_sm70_moe.py`, `test_skinny_moe_triton.py` (Experten-Kern).
- `tests/models/glm5next/test_moe_router_once.py`.
- Benchmarks: `benchmarks/benchmark_sm70_deepseek_v4_fused_router.py` (Last-CTA-Prototyp, reale Checkpoint-Gewichte, 40 Schichten, CUDA-Graph), `benchmarks/kernels/benchmark_glm_skinny_moe_routing.py` (Vorbild für bit-gleiche Hebel, echte Schichten und Routing), `benchmarks/kernels/benchmark_fused_topk.py`, `benchmark_sm70_router_native.py`, `benchmark_sm70_router_moe_chain.py` (Qwen).

## Tabelle: Kern im Profil → Python-Op → Quelle → Karten → Datentyp

Spalte Karten: V = Tesla V100 (Stufen 1–3), R = Quadro RTX 8000 (Stufen 0 und 4). "abgeleitet" heißt aus Code und Standardwerten erschlossen, Boot-Log oder Trace nicht gesehen.

| Kern im Profil | Python-Op | Quelle (Datei:Zeile) | Karten | Datentyp |
|---|---|---|---|---|
| `post_dot` (8,6 µs) | `mhc_fused_post_pre_tilelang`, bei 1–8 Zeilen nativ `sm70_glm_mhc_post_dot_q8_out` mit tile_n 6 (abgeleitet aus `mhc_fp32_stage` Default 1, `mhc_fused_post_dot` Default True) | `vllm/model_executor/kernels/mhc/tilelang.py:693`, `:825-830`, `:881-904`; `csrc/torch_bindings.cpp:1095`; Aufruf `model.py:962-980` | V, R | FP16-Aktivierung, FP32-Staging |
| `pre_norm` (19,6 µs) | `sm70_mhc_pre_norm_from_staging` → nativ `sm70_glm_mhc_pre_norm_out` bei 2–8 Zeilen (`mhc_native_verify` Default True, `threads=mhc_pre_threads` 1024; passt zu Gitter 6×1024) | `vllm/model_executor/kernels/mhc/triton.py:628-690`; `vllm/config/execution_policy.py:367-370`; `tilelang.py:1010-1040` | V, R | FP32 Staging → FP16 |
| Router `Kernel2` 13,4 + `splitKreduce` 4,0 | `F.linear` (cuBLAS-FP16-GEMM, Split-K), Tier 3 von `GateLinear` | `gate_linear.py:129-132`; `linear.py:472`; `utils.py:128` | V, R (gleicher Pfad) | Gewicht FP16 [256,4096], x FP16 [6,4096], Ausgabe FP16 |
| Router `unrolled` 8,4 | `output.to(torch.float32)` (vermutlich, nicht belegt) | `gate_linear.py:133-134` | V, R | FP16 → FP32 |
| Shared gate+up (31 µs) | `DeepseekV4MLP.gate_up_proj`, FP8 (QPN8) | `model.py:88-95`; Kerne `csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu` (V), `fp8_qpn8_mma_sm75.cu` (R); Zuordnung ANALYSE.md:77 | V, R | FP8-Gewicht (e8m0-Skalen), FP16-Aktivierung (Python-Verdrahtung nicht verfolgt) |
| act_and_mul (5 µs) | `SiluAndMulWithClamp` → `torch.ops._C.silu_and_mul_with_clamp` (Limit 10) | `vllm/model_executor/layers/activation.py:163-198`; `model.py:109-110` | V, R | FP16 |
| Shared down (18 µs) | `DeepseekV4MLP.down_proj`, FP8 | `model.py:96-104`, `:114-118` | V, R | FP8-Gewicht, FP16 |
| Top-k `topkGatingSoftplusSqrt` (12,8 µs) | `fused_topk_bias` → `ops.topk_hash_softplus_sqrt` → `_moe_C.topk_softplus_sqrt` | `fused_topk_bias_router.py:160-229`, `:106-144`; `_custom_ops.py:2607`; `csrc/moe/topk_softplus_sqrt_kernels.cu:82-426`, `:458-502`, `:591-593` | V, R (Hash-Zweig nur Stufe 0) | Logits FP32, Bias FP32; Ausgabe Gewichte FP32 [6,6], IDs int32 [6,6]; Hash: `tid2eid` int32 |
| Gruppenrouting (5,0 µs) | `group_routing` → Triton `_group_routing_kernel` | `sm70/routing.py:47-106`; Aufruf `skinny_sm70_moe.py:835` | V, R | int32 (perm [36], gids [36], goff [37]) |
| MoE-Up-Kern | `Mxfp4SkinnySm70Experts._grouped_projection` → `torch.ops._C.skinny_moe_qpn_sm70` | `skinny_sm70_moe.py:891-943` | V (M8N8K4), R (m16n8k8) | MXFP4-Gewicht + e8m0-Skalen, FP16-Aktivierung, Ausgabe FP16 [36, 4096] |
| MoE-Down + Gewichtsreduktion | gleiches Op mit `x_slot_major`, dann Triton `_mtp_weighted_reduce` | `skinny_sm70_moe.py:889`; `sm70/reduction.py:54-98` | V, R | FP16, Gewichte FP32 |

## Fusionsmöglichkeiten (mit vorhandenen Bausteinen)

Fest stehen drei Randbedingungen: M ist 5 oder 6 (CUDA-Graph-Größen), k = 6, 256 Experten, K = 4096, Gate-Gewicht 2,1 MB FP16.

A. Top-k-Epilog und Gruppenrouting in einen Kern verschmelzen (nur die Kette nach dem Gate). Beide haben nur ≤ 36 Slots. Bausteine: Auswahl- und Gewichtssemantik aus `topkGatingSoftplusSqrt` (Hash und Nicht-Hash), Sortierung aus `_group_routing_kernel`. Vorteil: spart einen Start und die Zwischenspeicherung (Ausgabe `topk_ids` und `topk_weights` bleiben, `token_expert_indices` entfällt). Kein Einfluss auf das Gate-GEMM, daher am wenigsten numerisches Risiko.

B. Gate-GEMM (FP16, 6 Zeilen) selbst als eigener Mehrzeilen-Kern statt cuBLAS-Split-K, mit FP32-Ausgabe direkt (spart `splitKreduce` und `unrolled`). Vorlage: `_sm70_dsv4_fp16_gemv_kernel` (`gemv.py:26-41`, eine Zeile je Programm) und das Qwen-Mehrzeilen-Gate (`qwen38_router_batch_sm70_out`). Wichtig: Gerundet wird heute auf FP16 vor dem Cast. Ein Kern mit FP32-Ausgabe ohne Rundung ändert die Logits leicht (Prototyp hat dafür `ROUND_TO_FP16`; 1Cat maß bei M=1 gleiche IDs auf 16 Seeds, größte normierte Gewichtsabweichung 3,61e-5). Auf der RTX gibt es dafür keinen vorhandenen Kern, nur Triton.

C. Vollfusion Gate + Top-k + Gruppenrouting (Last-CTA-Muster aus dem 1Cat-Prototyp, zu M=6 erweitert; `benchmarks/benchmark_sm70_deepseek_v4_fused_router.py:30-110`). Zusätzlich: Hash-Schichten 0–2 (Stufe 0) getrennt halten oder per `USE_HASH`-Zweig; Skalierung 1,5 und Normierung. Mehr Aufwand, höchstes Einsparpotenzial, aber 1Cat hat diesen Weg bei M=1 verworfen. Messen im 40-Schichten-Graphen mit allen Gate-Gewichten, nicht in einer Schicht (L2-Effekt).

D. Shared-Experten-Rechnung unabhängig davon (siehe Frage 3): kein Hindernis, aber Entscheidung "Stream bleibt aus" gilt.

E. Nebenbefund: Die FP13-Puffer (laut `VRAM_FUER_TEMPO.md` etwa 105,6 MiB je V100-Stufe) werden im 5/6-Zeilen-Decode nicht gelesen. Ob sie für Einzeilen-Schritte gebraucht werden, ist offen.

## Offene Punkte (nur Profil oder Test klärt das)

1. Echte Kernnamen und Gitter der drei Gate-Kerne am Aufrufort: nsys-Auszug mit Stack oder `torch.profiler` für eine MoE-Schicht. Bestätigt `Kernel2`/`splitKreduce` als cuBLAS-Split-K des Gates und `unrolled` als FP16→FP32-Cast.
2. Warum `unrolled` 8,4 µs braucht und ob es nur der Cast ist.
3. Ob `Mxfp4SkinnySm70Experts`, `moe_backend=auto` und der MHC-Native-Pfad (post_dot/pre_norm) in der Produktion wie abgeleitet gewählt sind: Boot-Log (`moe_kernel_selections`) und Trace.
4. Ob es in der Produktion Einzeilen-Schritte gibt (und wie oft), in denen das FP13-GEMV läuft.
5. Wie viel von den 25,8 µs (V100: 13,4+4,0+8,4) auf der V100 und der RTX Bytezeit gegen Latenz ist. Gate-Gewicht 2,1 MB, Bytezeit bei 85 % etwa 2,7 µs (V100) beziehungsweise 3,7 µs (RTX), das sind rein rechnerische Werte.
6. Wie weit die Top-k-Toleranz (IDs exakt, Gewichte lose im Test) bei Rundungsunterschieden aus Gate-Fusion und `__logf`/`__expf` gegen `tl.log`/`tl.exp` hält: nötig ist ein Test mit M=5 und 6, 256 Experten, k=6, Hash und Nicht-Hash, echten Gate-Gewichten, und Gleichstand-Fällen (niedrigster Index gewinnt).
7. Ob die DSpark-Entwurfsschichten (Stufe 4) denselben Router-Pfad mit 5 Zeilen nutzen und ob dort Hash-Schichten vorkommen: `dspark.py` nicht verfolgt.
8. Ob v15 (Produktions-PYTHONPATH) an allen zitierten Stellen mit v16 übereinstimmt.

## Ergebnis 25.11.2026: schmaler FP16-Linear-Kern (Zweig `dsv4-router-fuse` 779badd38 + 3c7b50cd8, im Fork): nicht übernommen

Kern, Regelkacheln, Einbau an `UnquantizedLinearMethod` (opake Op, Schalter `kernel_config.skinny_fp16_linear` im Compile-Hash), DeepSeek-Gate ohne FP32-Cast.
Tests 208/208 auf RTX und V100. Mikrobenchmark (`~/.cache/bench-scripts/skinny_2026-11-25/`, `skinny2b_2026-11-25/`): mit bester Kachel 1,02–1,26× (RTX),
1,13–1,45× (V100, `weights_proj` 5,6×) gegen cuBLAS; Router-Kette je Schicht RTX 22,0 → 19,1 µs, V100 25,3 → 19,2 µs, Top-k-IDs 320/320 gleich.
A/B DeepSeek `ab_skinny2_dsv4_2026-11-25c.log` (T T P T P, T1 kompiliert): Runde greedy 48,3 gegen 48,7 ms, AIfred-Sampling 48,7–48,8 gegen 49,2 ms (−0,9 %);
tok/s je nach Annahme +2,2 % (AIfred) bzw. −3,1 % (greedy), Decode kurz −0,6 %, GSM8K-Decode −0,4 %, 29 richtig, Prefill gleich, Langkontext je Schritt
−0,5 bis −1,2 %; Texte 8/8 anders formuliert (nicht gelesen). Also ≈ 0,45 ms je Runde statt der überschlagenen 0,9 ms. GLM-A/B abgebrochen.
Peuqui 25.11. ~04:30: „wegen der kleinen Gewinne lohnt es sich nicht wirklich“. Der bitgleiche Teil (FP16-Logits direkt in den Top-k, ≈ 4 µs je Schicht,
Nicht-Hash-Schichten) kann mit einem ohnehin anstehenden Bündel mitfahren.
Lehre: Die Kernzeiten aus nsys (Gate 13,4 + 4,0 + 8,4 µs) lagen weit über den Zeiten im CUDA-Graphen ohne Profiler (8,9 + 2,5 µs); Hebel-Schätzungen aus
dem Profil vor dem Bau im Mikrobenchmark nachmessen.
