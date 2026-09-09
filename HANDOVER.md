# Übergabe — Stand 09.09.2026 mittags

**Betriebsstand steht in `STAND.md`. Damit anfangen, nicht mit diesem Dokument.**
Hier steht nur, was als Nächstes ansteht und was du über den letzten Tag wissen
musst, um nicht dieselben Wege noch einmal zu gehen.

---

## Auftrag: DFlash2 auf Turing messen

Die Frage: Bringt DFlash2 gegenüber MTP nochmals Tempo? MTP liefert beim 27B
eine Annahmelänge von 2,963 bei k=3. DFlash2 arbeitet mit einem eigenen
Entwurfskopf über mehrere Zielschichten und Blockgröße 8 statt eines einzelnen
MTP-Blocks, hat also strukturell mehr Spielraum.

**Was bereitliegt:**
- Entwurfskopf heruntergeladen: `incoai/Qwen3.8-27B-DFlash2`, 3,6 GiB,
  `DFlash2DraftModel`, `block_size 8`, `target_layer_ids [5,19,33,47,61]`,
  dtype bfloat16.
- Messvorrichtung `tools/mtp-diagnostics/speed_dflash.sh` — **noch nie
  gelaufen**, also erst gegen eine bekannte Konfiguration verifizieren, bevor
  du ihren Zahlen glaubst.
- Vergleichsmaßstab: `speed_27b.sh <name> fork 3` liefert heute 73,35 tok/s bei
  Annahmelänge 2,963, Text-SHA `0106659946c064b1`.

**Vorbehalt:** 1Cat hat auf SM70 mehrere offene DFlash2-Baustellen (#561, #405,
#547). Das Terrain bewegt sich; vor dem Start `git fetch` und die offenen PRs
ansehen.

---

## Was gestern/heute passiert ist — die Kurzfassung

**Turing fährt seit heute den gemeinsamen GDN-Pfad.** Der fork-eigene
sm75-Zweig ist raus, samt beider sm75-Dateien und des toten Apparats
(`c86fc8d`, `ac0b0ae`, `e529348`, `afc22a6`). Kosten 1,9 % beim 27B, bei
Flash-Next unter der Rauschgrenze. Verifiziert für beide Modelle auf dem
Endstand — Zahlen und Belege in `STAND.md`, Abschnitt „Messstand 09.09.".

**Overlay und Deployment sind wieder deckungsgleich** (0 von 94). Sie waren
auseinandergelaufen: die venv trug Fix 2, 3 und 5, das Overlay nicht. Wer
direkt in die venv patcht, muss zurückschreiben — sonst löscht der nächste
Deploy die Arbeit.

**Vier PRs bei 1Cat offen und CI-grün:** #572 (Turing bootfähig), #573
(Qwen4Exp-MTP stufenlokal unter PP), #574 (Output-Trim auf allen PP-Rängen),
#576 (SM70-Quant-Gate liest das eigene Gerät). `pre-run-check` ist jetzt grün,
weil die Autorenschwelle von vier gemergten PRs erfüllt ist. Keine Reaktionen
bisher.

---

## Wo du NICHT weitersuchen solltest

**Der Restabstand von 1,74 % sitzt nicht im GDN-Eingangssplit.** Die frühere
Zuordnung zu `_sm70_compile_graph_slice_dim` war ein Denkfehler: wenige Zeilen
darunter stehen `z.contiguous()`, `b.contiguous()`, `a.contiguous()` — nach
`index_select` No-ops, mit einem View kopieren sie dort erst recht. Die Kopie
verschwindet nie, sie wandert nur. Vier Messungen, **jede Sparmaßnahme
langsamer**; `a` als View ist korrekt, kostet aber 0,4 %. Herleitung mit allen
Zahlen: `STAND.md` Punkt 5 und die Gedächtnisnotiz
`project_z_slice_materialization_closed`. Die widerlegte Kernel-Rechnung samt
Beweisführung steht im vorigen Übergabestand (`git show ac9d517:HANDOVER.md`).

Wenn der Abstand jemanden interessiert: Kandidaten wären die übrigen
Unterschiede des entfernten sm75-Pfads — eigene Kern-Op, Faltungsbehandlung,
Norm-Fusion. Nicht die Eingangsprojektion.

---

## Offene Fäden

1. **DeepSeek-V4 nach dem GDN-Umstieg nicht nachgemessen.** Strukturell nicht
   betroffen, aber das ist ein Argument und keine Messung. Siehe `STAND.md`,
   offener Punkt 3.
2. **PLE-Überlaufkaskade** (vier Stufen) — unverändert offen, `STAND.md`
   Punkt 6.
3. **Zwei Upstream-Änderungen beim nächsten Wheel mitziehen:**
   `custom_all_reduce.py` und `platforms/cuda.py`, beide UUID-GPU-Auswahl.
   Heute ohne Wirkung für uns. Hinweise stehen in `fork_patches_150/STATUS.txt`.
