# KDA-Ausgangsnorm: welchen Weg nimmt `self.o_norm` auf main 7db135fa0 und in unserer Produktion?

Stand 25.11.2026, nur gelesen und gezählt, kein Umbau. Bäume: main = `1Cat-vLLM-prmain` (7db135fa0), Produktion = `1Cat-vLLM-v16` (e336f616e).

## Ergebnis in Kürze

1. Standard auf main und in der Produktion: `custom_ops = [..., "none"]`. `FusedRMSNormGated` ist damit nicht eingeschaltet und läuft als `forward_native`, eager (nicht kompiliert), im CUDA-Graphen aufgezeichnet. Das ist der Weg, den unser Kern ersetzt.
2. Kernel je Schicht und Schritt (CPU-Zählung der Torch-Operatoren, jeder ist auf CUDA ein Kernel): 11 beim fp32-Eingang (fp32-Zweig), 12 beim fp16-Eingang (alles andere, also die RTX auf main und alle Nicht-recurrent-Schritte). GLM-5.3 hat 34 KDA-Schichten.
3. Der fp32-Zweig gilt auf main nur für Capability (7, 0); die RTX (7, 5) läuft auf main im fp16-Zweig, ebenfalls mit dem eager-`forward_native` (12 Kernel). In der Produktion (v16) steht `in ((7, 0), (7, 5))`: dort nehmen beide Kartentypen den fp32-Zweig.
4. `forward_cuda` (Triton `rms_norm_gated`) läuft nur, wenn die Op eingeschaltet ist: ausdrücklich per `+fused_rms_norm_gated` oder weil `custom_ops` "all" enthält (Kompilierung aus, siehe unten). Beides trifft in der Produktion nicht zu.

## 1. Welcher `custom_ops`-Wert gilt

- `vllm/config/vllm.py:1507-1514`: enthält die Liste weder "all" noch "none", wird "none" angehängt, wenn `backend == "inductor"` und `mode != NONE`, sonst "all".
- `vllm/model_executor/custom_op.py:272-289` (`enabled`) und `291-304` (`default_on`): eingeschaltet nur, wenn "all" gilt oder `+<name>` in der Liste steht, und kein `-<name>`.
- `vllm/model_executor/custom_op.py:174-205` (`dispatch_forward`): nicht eingeschaltet gibt `maybe_compile(self.forward_native, enable=compile_native)` zurück. `FusedRMSNormGated` ruft `super().__init__()` ohne Argumente (`vllm/model_executor/layers/fla/ops/kda.py:514`), also `compile_native=False`: `forward_native` läuft eager. Die Wahl wird in `__init__` festgeschrieben (`custom_op.py:133`).
- Nirgends in `vllm/` wird `+fused_rms_norm_gated` gesetzt; Treffer nur in Tests (`grep -rn fused_rms_norm_gated vllm tests`: `tests/models/glm5next/test_kda_glue.py:33`, `test_kda_norm_dispatch.py:24,38`).

## 2. Ist Inductor/Compile-Modus aktiv?

Standard (V100 oder RTX): `vllm/config/vllm.py:1494-1498` setzt `mode = VLLM_COMPILE`, wenn nichts anderes gesetzt ist und das Optimierungslevel über O0 liegt; CUDA-Standard-Backend ist Inductor. Der SM70-Flash-V100-Baseline in `vllm/platforms/runtime_defaults.py:395-399, 495-530` setzt bei mindestens einer V100 `VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1`, `mode = VLLM_COMPILE` und `cudagraph_mode = FULL_AND_PIECEWISE`. Nur diese Fälle schalten Kompilierung ab und damit `custom_ops` auf "all":
- `enforce_eager`, `TORCH_COMPILE_DISABLE=1` oder `VLLM_USE_BREAKABLE_CUDAGRAPH` (`runtime_defaults.py:313-337`): `mode = NONE`.
- `VLLM_SM70_FLASH_V100_DECODE_GRAPH_NO_COMPILE=1` (`runtime_defaults.py:690-715`, Standard False laut `vllm/envs.py:996`): `mode = NONE`.

Bei `mode = NONE` ergibt `vllm.py:1507-1514` "all", `FusedRMSNormGated` ist eingeschaltet, `forward_cuda` läuft. Das ist der Hintergrund von 60dd820a7 / #1165, die davon ausgehen, dass der SM70-Decode-Graph nicht kompiliert ist.

Wichtig für die Wirkung: Der Modus VLLM_COMPILE heißt nicht, dass `Glm5NextModel` kompiliert wird. `vllm/models/glm5next/` enthält keinen `@support_torch_compile`; beim Boot erscheint je Worker die Warnung "`torch.compile` is turned on, but the model ... does not support it" (`vllm/config/vllm.py:2646-2658`). Die KDA-Schicht läuft also eager, auch wenn "none" und Inductor gelten; "none" bedeutet dann nur: die Op ist aus, `forward_native` läuft eager.

## 3. Unsere Produktion (GLM-5.3-Flash-320B-A18B-NVFP4-DFlash2-vllm, Baum v16)

- llama-swap-Eintrag (`/home/mp/.config/llama-swap/config.yaml`, nur gelesen): `--compilation-config '{"cudagraph_capture_sizes":[7],"max_cudagraph_capture_size":7}'`, kein `custom_ops`, kein `mode`, kein `enforce_eager`; `PYTHONPATH=.../1Cat-vLLM-v16`; `--dtype half`; Karten 3x V100 + 2x Quadro RTX 8000 (PP5).
- Gemessener Bootzustand (Journal `/home/mp/.cache/bench-scripts/ab_kvb_glm_2026-11-24a/journal-1-T.txt`, Zeile 128, Engine-Konfiguration): `'mode': VLLM_COMPILE`, `'backend': 'inductor'`, `'custom_ops': ['+quant_fp8', 'none']`, `'cudagraph_mode': FULL_AND_PIECEWISE`, `cudagraph_capture_sizes: [7]`, `'use_inductor_graph_partition': False`. Dieselbe Datei, Zeilen 222-238: Warnung "torch.compile is turned on, but the model ... does not support it" für alle fünf Worker. (Dieses Journal ist vom Lauf des 24.11. aus einem v15-/v16-nahen Stand; die Konfigurationslogik ist in v16 und main an diesen Stellen gleich.)
- Ergebnis: `fused_rms_norm_gated` ist ausgeschaltet (`none`, kein `+`), `FusedRMSNormGated.forward_native` läuft eager im aufgezeichneten FULL-Decode-Graphen. Der v16-Baum ist an den betroffenen Stellen unverändert gegenüber main (`git diff pr-base-7db135fa0 prod-v16`: `fla/ops/kda.py` und `custom_op.py` gleich; `config/vllm.py` nur 3 Zeilen anders, `custom_ops.append("none")` weiter bei Zeile 1512).
- Fp32-Zweig in v16: `vllm/models/glm5next/nvidia/kda.py:382-386`, `get_device_capability() in ((7, 0), (7, 5))`; also nehmen V100- und RTX-Stufen den fp32-Zweig. Ohne unseren Kern liefen dort 11 eager Kernel je Schicht; mit ihm (`kda.py:694`, direkter Aufruf `kda_gated_rmsnorm`) 1.

## 4. Kernelzahl (gezählt, nicht gemessen)

Zählung per `TorchDispatchMode` auf CPU mit `FusedRMSNormGated(128, activation="sigmoid", dtype=float16)` unter `custom_ops=["none"]` (Skript `/tmp/claude-1000/-home-mp-Projekte-vllm-research/d820eb98-aa5b-4b03-9979-706846eee66f/scratchpad/count_ops.py`); die Bindung zeigte `forward_native`. Operatorfolge, je Operator ein Kernel auf CUDA:

- fp32-Eingang (fp32-Zweig, 11): `pow`, `mean`, `add`, `rsqrt`, `mul`, `_to_copy` (Gewicht in fp32), `mul`, `_to_copy` (Gate in fp32), `sigmoid`, `mul`, `_to_copy` (zurück nach fp16).
- fp16-Eingang (Nicht-fp32-Zweig, 12): dieselben, davor ein zusätzliches `_to_copy` (`x.float()`).
- `forward_cuda` (nur wenn eingeschaltet): `rms_norm_gated` -> `layer_norm_gated_fwd`, ein Triton-Kernel (`fla/ops/kda.py:462-499`); die Startzahl auf CUDA ist nicht gezählt.
- Unser Kern: 1 Start (`kda_glue.py:77-105`).

Die Kette läuft in jedem Schritt, auch bei Prefill-/Chunk-Schritten (dort fp16-Eingang, 12 Kernel; bei vielen Zeilen fällt das nicht ins Gewicht).

## 5. Antwort auf "gibt es den eager 11-Kernel-Weg auf main, auf welchen Karten?"

Ja, standardmäßig auf allen Karten, solange Inductor und ein Compile-Modus gelten (V100 wie RTX, auch bei einem Modell, das gar nicht kompiliert wird). Auf main: V100 im fp32-Zweig (Recurrent-Schritte, 11 Kernel); V100 in Prefill/Chunk und RTX in allen Schritten im fp16-Zweig (12 Kernel). Nicht betroffen sind nur Rigs mit `mode = NONE` (dort `forward_cuda`, Triton). Für die Produktion gilt dasselbe, mit dem Unterschied, dass dort die RTX durch das erweiterte Capability-Gate ebenfalls im fp32-Zweig liegt (11 Kernel).

## 6. Vorschlag: Einhängung passend zu 1Cats CustomOp-Vertrag (#1165)

Der Vertrag (Kommentar an `self.o_norm(...)` in `kda.py` auf main; `test_kda_norm_dispatch.py`): die Op entscheidet, nicht der Aufrufer. Ein ausdrückliches `-fused_rms_norm_gated` bleibt `forward_native`, ein `+fused_rms_norm_gated` wird `forward_cuda`; die Wahl wird im Konstruktor festgeschrieben. Unser Kern darf diesen Vertrag nicht umgehen. Es passen zwei Stufen:

1. Der Kern wird Teil der Op, als Rumpf von `FusedRMSNormGated.forward_cuda`. Er ist bitgleich zu `forward_native` (`kda_glue.py:1-20`, geprüft durch `test_kda_glue.py`) und erfüllt damit auch, was `test_sm70_kda_output_norm_triton_matches_decomposed` (`test_sm70_kda.py:276-313`) von `forward_cuda` verlangt. `forward_cuda` ruft ihn bei Sigmoid-Aktivierung, Breite 128, ohne `residual`/`prenorm` und mit Gewicht; alle anderen Aufrufe laufen wie bisher durch `rms_norm_gated`. Das ist eine ausdrückliche Verzweigung nach Eignung (Vorbedingung des Kerns), kein stiller Fallback.
2. `kda.py` ruft in beiden Zweigen wieder `self.o_norm(...)` auf, der direkte Aufruf `kda_gated_rmsnorm` entfällt. Der Kern wirkt dann nur, wenn die Op eingeschaltet ist. Eingeschaltet wird sie wie andere Ops auch: entweder per Aufruf (`--compilation-config '{"custom_ops":["+fused_rms_norm_gated"]}'` im llama-swap-Eintrag; das ergänzt den Standard "none", wie das `+quant_fp8` der Produktion aus `vllm/config/vllm.py:1479-1481`), oder als Plattformvorgabe in `vllm/config/vllm.py` nach dem Muster `if "-<op>" not in custom_ops: custom_ops.append("+<op>")` (Zeilen 1479-1481 für `quant_fp8`, 1782-1783 für `rms_norm` bei PP>1). `enforce_enable=True` am Konstruktor wäre die falsche Wahl: es überstimmt ein ausdrückliches `-fused_rms_norm_gated` und bricht damit #1165.

Zur RTX (7, 5): Bei dieser Einhängung hängt der Gewinn nicht mehr am Capability-Gate. Der Kern lädt bereits mit `.to(tl.float32)` (`kda_glue.py:56-68`); die einzige Sperre ist `assert x.dtype == torch.float32` (`kda_glue.py:89`). Fällt sie, ist der Kern auch für fp16-Eingang bitgleich zu `forward_native` (das beginnt ebenfalls mit `x.float()`), und die RTX auf main im fp16-Zweig bekäme 12 -> 1 Kernel ohne `(7, 5)` im fp32-Gate. Ob die Recurrent-Ausgabe auf Turing in fp32 bleibt, ist eine eigene Präzisionsentscheidung und gehört nicht in diesen PR (im Fork `kda.py:382-386`). Voraussetzung: `test_kda_glue.py` um fp16-Fälle und den Abgleich gegen `-fused_rms_norm_gated` ergänzen.

Zu "nur der Packteil": Der Packteil (`kda_pack_recurrent_inputs`, `_recurrent_inputs`) hängt an keiner Op und keinem Gate und gilt für alle Karten; er allein ist ohne Vertragsfrage sendbar (Ersatz von vier `.contiguous()`-Kopien). Die Norm verlangt dagegen die Op-Einhängung oben, sonst bleibt der PR bei der Umgehung, die die Prüfung am 23.11. beanstandet hat. Auch mit Einhängung wirkt die Norm-Stufe auf main erst, wenn jemand `+fused_rms_norm_gated` setzt, weil der Standard "none" bleibt. Offen für dich: nur Konfigurationseintrag (kein Verhaltenswechsel auf main) oder Plattformvorgabe (Verhaltenswechsel für alle GLM-5.3-Nutzer auf main).

## Offene Punkte

- Die Zählung (11/12) kommt aus den Torch-Operatoren auf CPU, nicht aus einem CUDA-Profil. Sie deckt sich mit dem Docstring von `kda_glue.py` ("eager chain, 11 kernels").
- Nicht per Profil belegt: dass die RTX auf main im Decode-Graphen tatsächlich den fp16-Zweig (12 Kernel) fährt; aus dem Code folgt es (`_use_sm70_fp32_recurrent_output` nur (7, 0), sonst `self.o_norm(core_attn_out, g2)`), gelaufen ist es nicht.
- Das Journal mit der Engine-Konfiguration stammt von einem A/B-Boot am 24.11. (Stand nahe v16). Ein Boot gerade auf `1Cat-vLLM-v16` mit `custom_ops` im Log habe ich nicht gelesen; die Konfigurationslogik ist dort gleich.
