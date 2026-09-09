# Übergabe — Stand 09.09.2026 abends

**Betriebsstand steht in `STAND.md`. Damit anfangen, nicht mit diesem Dokument.**
Hier steht nur, was als Nächstes ansteht und was du über den letzten Tag wissen
musst, um nicht dieselben Wege noch einmal zu gehen.

---

## Auftrag: sm75-Variante der Skinny-Kernel

**DFlash2 läuft und schlägt MTP** (V100: 74,09 gegen 66,13 tok/s). Offen ist
nur noch, dass Turing dabei hinter die V100 zurückfällt: 69,13 gegen 74,09.

Der Rückstand ist **vollständig lokalisiert** — 435 ms in zwei Kerneln, siehe
`STAND.md` Punkt 8 mit den nsys-Zahlen. Ursache: `kernels/skinny_kernels.cu`
hat **keine einzige `__CUDA_ARCH__`-Fallunterscheidung** und nutzt durchgängig
Voltas `mma.sync.aligned.m8n8k4`. Turing führt die Instruktion aus, ist aber
auf `m16n8k8` ausgelegt.

**Reihenfolge:**
1. Fragment-Layouts für `m16n8k8` auf Turing bestimmen. Das Werkzeug existiert:
   `mma8_probe.cu` hat dasselbe seinerzeit auf der V100 gemacht (Kommentar bei
   Zeile 609 in `skinny_kernels.cu` nennt die abgeleiteten Maps).
2. Architekturabhängige Variante des MMA8-Pfads, gegen den bestehenden
   Volta-Pfad abgesichert.
3. Dasselbe für `skinny_fp8_qpn8` — er stellt mit 217 ms etwa die Hälfte des
   Rückstands.

**Erwartung ehrlich halten:** Die 435 ms zu schließen bringt Gleichstand mit
der V100, nicht mehr. Für ein echtes Übertrumpfen müsste der Turing-Kernel
besser sein als Voltas; möglich, weil `m16n8k8` mächtiger ist und die RTX bei
MTP schon 11 % vorn liegt. Oberes Ende wären die ~82 tok/s aus dem
MTP-Verhältnis.

---

## Wo du NICHT weitersuchen solltest

Vier Erklärungen für die Turing-Lücke sind **gemessen widerlegt**:

- **QPN8-Rerank für den Kandidaten-TopK.** War mein Verdacht aus dem Code. Im
  Decode-Profil taucht kein `topk`/`sort`-Kernel in den Top-14 auf.
- **Das AllReduce.** 1.697 ms auf Turing gegen 1.530 auf V100 bei nahezu
  gleicher Aufrufzahl — beide Kartenpaare hängen identisch an (OCuLink,
  Gen3 ×4, kein P2P). Grundlast, kein Differenzierer (Peuqui).
- **Die Compile-Vorgaben** (`fuse_norm_quant`, `rms_norm=['vllm_c']`, per
  `SM70TUNE=1` in `speed_dflash.sh`): 69,13 → 69,45 tok/s, also +0,5 %.
  Bemerkenswert nur, dass die Annahmelänge dabei exakt den V100-Wert trifft.
- **Marlin als Ersatz für Skinny:** 43,87 gegen 69,13 tok/s, also 37 %
  langsamer. Unser Skinny-Kernel ist auf Turing bereits die beste Route.

**TurboMind ist kein Rebuild, sondern eine Portierung.** Der Quelltext hat zwar
Turing-Instruktionen und eine `config_sm75_s16816.h`, aber es existieren **nur
Volta-Kernel-Instanzen** (`sm70_884_4/8/16.cu`). `vllm/_C.abi3.so` trägt 39
ELF-Einträge, alle `sm_70`, kein PTX — das Gate in `sm70_turbomind.py` zu
öffnen brächte nichts.

---

## Offene Fäden

1. **QUASAR-QAT ist in diesem Stack unbrauchbar** — `STAND.md` Punkt 7.
   1Cats Referenzcheckpoint degeneriert bei uns (derselbe Nachsatz an jedem
   Satz, dann Abbruch ohne Antwort). Ausgeschlossen sind DFlash2, Chat-Template,
   Kontextlänge und Sampling. Verbleibender Verdacht: der
   `compressed-tensors`-Pfad. Auf Turing lädt er ohnehin nicht
   (`gptq_marlin_repack` verlangt Vielfache von 64, eine Schicht hat 8.240).
   **Gemessen wird auf RadixArk.**
2. **DeepSeek-V4 nach dem GDN-Umstieg nicht nachgemessen** — `STAND.md`
   Punkt 3, unverändert offen.
3. **PLE-Überlaufkaskade** (vier Stufen) — `STAND.md` Punkt 6, unverändert.
4. **`prof_prefill.sh` ist nicht lauffähig** — `STAND.md` Punkt 9. Es übergibt
   `--output` an `nsys launch`; die Option gehört an `nsys start`. In
   `prof_dflash.sh` korrigiert, dort nicht.
5. **Zwei Upstream-Änderungen beim nächsten Wheel mitziehen:**
   `custom_all_reduce.py` und `platforms/cuda.py`, beide UUID-GPU-Auswahl.
   Hinweise in `fork_patches_150/STATUS.txt`.
6. **Die beiden DFlash2-Patches sind PR-Kandidaten für 1Cat.** Der Guard macht
   DFlash2 mit jedem Checkpoint fahrbar, dessen LM-Head mitquantisiert ist
   (bei ModelOpt-Exporten der Normalfall); das BF16-Gate bringt Turing die
   Range-Erhaltung. Beide sind mit byteidentischem Text auf zwei Architekturen
   belegt. Vor einer Meldung: `AGENTS.md`-Pflichtregeln beachten.

---

## Vier Fallen, die an einem Tag je einen Lauf gekostet haben

Alle vier stehen ausführlich in `STAND.md`, Abschnitt „Fallstricke":

- Ein fremdes Profil wörtlich zu übernehmen stellt **nicht** seine Bedingungen
  her — 1Cats Zeile hat kein `--disable-custom-all-reduce`, weil ihre V100 P2P
  können. Bei uns warten dann beide TP-Ränge ewig im Allreduce.
- `utilization.gpu` ist keine Fortschrittsanzeige: 100 % bei 46 W heißt
  Leerlauf. Leistungsaufnahme messen, und `py-spy dump` in mehreren Proben.
- vLLM liefert den Denkblock im Feld `reasoning`, nicht `reasoning_content`.
  Rohantwort immer als JSON mitschreiben.
- Ein Qualitätsurteil braucht die volle Ausgabelänge, und eine auffällig
  **hohe** Annahmelänge ist ein Warnsignal: eine Wiederholungsschleife ist
  trivial vorhersagbar (gemessen 5,569 von 8 bei völlig degeneriertem Text).
