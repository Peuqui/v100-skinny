# Rebase 3 auf 1Cat main 73da3445d (#1616): offene Fragen

Zweige: pr-mrv2-dspark-r1125 (PR #1221, Kopf d9ad661c6), pr-adaptive-verify (Kopf aeeac556d). Alte Köpfe: f01e1c581, 4018ddb54.

1. `_add_dspark_width_graphs` (main, vllm/v1/worker/gpu/cudagraph_utils.py) greift bei JEDEM method=="dspark", nicht nur bei Inkling2.
   Es hängt im Legacy-Zweig von `CudaGraphManager.__init__` (der Split-MTP-Zweig gilt nur für method "mtp"), also auch für unsere
   DeepSeek-DSpark-Einträge (k=5 bzw. k=7, max-num-seqs 1): zusätzliche Ziel-Decode-Graphen für die Abfragelängen 1 bis k (je ein
   Graph je Länge bei max_num_reqs 1), die bei fester Entwurfsbreite nie abgespielt werden. Kostet Aufnahmezeit und Graph-Pool-Speicher
   auf den knappen Karten. Vorschlag: die Funktion mit `self.vllm_config._is_inkling2_dspark()` begrenzen (eine Zeile in mains Funktion).
   Nicht umgesetzt, weil es eine main-Zeile ändert und nicht beauftragt war. Zu entscheiden: in PR #1221 mitführen oder als eigener
   kleiner PR an 1Cat (Inkling2-Autor), oder im Fork vor prod-v17 lokal.
2. Die Hooks sind auf main 73da3445d selbst rot (kein Fehler unserer Zweige):
   - check-layering: #1616 hat die Baseline nicht gepflegt (Inkling2-Dateien, cuda_communicator, block_table, cudagraph_utils u. a.);
   - check-architecture-reference / test-architecture-reference: runtime_reference.md ist auf main veraltet (zwei Zeilen: Zeilennummern in
     speculative.py und VllmConfig.__post_init__); die Neuerzeugung auf main und auf beiden Zweigen ergibt für diese zwei Zeilen dasselbe;
   - mypy-3.10: main hat 49 Fehler in 15 Dateien (u. a. model_runner.py try_sample_target, async_scheduler.py DSpark-Zeilen).
   Ich habe nichts davon repariert (würde Zeilen von #1616 ändern und die Zweige mit fremden Korrekturen vermischen). Rückfrage: einen kleinen
   Reparatur-PR an 1Cat vorschalten, oder die Reparatur in unsere Zweige falten?
3. PR-Text zu #1221: Commit 1 sagt "Admits method 'dspark' on MRV2"; main lässt dspark seit #1616 für Inkling2 zu. Der Text sollte ergänzen,
   dass `init_speculator` Inkling2-Entwürfe weiter an `Inkling2DSparkSpeculator` leitet und der generische Blockentwurf nun in
   dspark/block_drafter.py liegt (Pfad gegenüber prod-v16 geändert: dort dspark/speculator.py). Commit-Texte nicht angefasst.
4. Fork-Folge: prod-v16 enthält unseren Speculator unter dspark/speculator.py. Beim nächsten Merge von 1Cat main in den Fork kollidiert die
   Datei mit mains Inkling2-Speculator; Auflösung wie hier (unsere Klasse nach block_drafter.py, mains Datei unverändert).
5. `init_speculator` ruft `vllm_config._is_inkling2_dspark()` (private Methode) aus einem anderen Modul. Gewollt wiederverwendet statt
   dupliziert; falls 1Cat das lieber öffentlich hätte, wäre das eine Umbenennung in mains Zeilen.
6. Der Baum 1Cat-vLLM-prmain2 steht jetzt auf 73da3445d (losgelöst), wie beauftragt; vorher 28f321c2c. Während meiner Läufe lief dort kurz ein fremder
   run_pytest_per_file (Ausgabe t3b_main); er war bei meinem letzten Blick beendet. Bitte den Baum bei Bedarf selbst zurückstellen.
