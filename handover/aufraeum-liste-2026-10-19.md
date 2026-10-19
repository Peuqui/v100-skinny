# Aufräumliste 19.10.2026 — ausgeführt 19.10. nachmittags (Freigabe Peuqui), Punkt 6 (lose Blobs) NICHT

Ergebnis: Punkte 1–5 und 7 gelöscht bzw. entfernt (43 Arbeitsbäume ohne Fehler, 6 Einträge per prune), frei 134G → 236G (85 % → 73 %). Produktion importiert, Zweige der entfernten Bäume vorhanden, kein Docker-Befehl, kein Container-Einhängepunkt berührt.

**Punkt 6 (lose Blobs) bleibt:** Der Container stt-suite bindet ~/.cache/huggingface ein, und die Blobs sind ONNX-Sprachmodelle (audio_signal, LSTM-Decoder, Vorverarbeitung; vermutlich CPU-Parakeet), auf die kein Symlink zeigt. Punkt 8 (DFlash2-Kopf) bleibt für Planpunkt Prompt-Lookup (1Cats ngram_assist gibt es nur für DFlash2).

Platte vorher: 736G belegt, 134G frei (85 %). Geprüft: keine Symlinks/Editable-Installs/llama-swap-Pfade aus Betrieb, Bau-venv oder Haupt-Checkout zeigen auf einen Kandidaten; alle Detached-Commits der Kandidaten von Zweigen/Tags erreichbar; Zweige und Tags bleiben.

## 1. ~/.cache/vllm-forkmain (Compile-Cache der A/B-Fork-Arme, zuletzt 18.10.) — 30691 MiB
rm -rf /home/mp/.cache/vllm-forkmain

## 2. Alte Compile-Hashes in ~/.cache/vllm/torch_compile_cache — behalten: f753253985 e5dbcdedfb 86ff3a79c6 6d9968726f 429a9df94b (seit 18.10. 21:40 benutzt, 1007 MiB)
/home/mp/.cache/vllm/torch_compile_cache/01e0efa32f
/home/mp/.cache/vllm/torch_compile_cache/0251396ad5
/home/mp/.cache/vllm/torch_compile_cache/05aa6fd0bf
/home/mp/.cache/vllm/torch_compile_cache/09e9557cee
/home/mp/.cache/vllm/torch_compile_cache/0a6fac61f2
/home/mp/.cache/vllm/torch_compile_cache/0c03f174b7
/home/mp/.cache/vllm/torch_compile_cache/0e8822d4ec
/home/mp/.cache/vllm/torch_compile_cache/13df49c000
/home/mp/.cache/vllm/torch_compile_cache/15bde70900
/home/mp/.cache/vllm/torch_compile_cache/1670e980cb
/home/mp/.cache/vllm/torch_compile_cache/171804484e
/home/mp/.cache/vllm/torch_compile_cache/17d6341cfe
/home/mp/.cache/vllm/torch_compile_cache/1879ec8c52
/home/mp/.cache/vllm/torch_compile_cache/18fa3fabf5
/home/mp/.cache/vllm/torch_compile_cache/2129fedf49
/home/mp/.cache/vllm/torch_compile_cache/23056ad0a8
/home/mp/.cache/vllm/torch_compile_cache/255ee94013
/home/mp/.cache/vllm/torch_compile_cache/256161b9c7
/home/mp/.cache/vllm/torch_compile_cache/26b65152eb
/home/mp/.cache/vllm/torch_compile_cache/26ec165497
/home/mp/.cache/vllm/torch_compile_cache/28ff63f779
/home/mp/.cache/vllm/torch_compile_cache/2a2fd47123
/home/mp/.cache/vllm/torch_compile_cache/2bb4a39ffd
/home/mp/.cache/vllm/torch_compile_cache/2c2e32eea0
/home/mp/.cache/vllm/torch_compile_cache/2cb117e37a
/home/mp/.cache/vllm/torch_compile_cache/31fd415d80
/home/mp/.cache/vllm/torch_compile_cache/32f1fdbe57
/home/mp/.cache/vllm/torch_compile_cache/3367dead41
/home/mp/.cache/vllm/torch_compile_cache/3400f872cf
/home/mp/.cache/vllm/torch_compile_cache/34c56c7eb9
/home/mp/.cache/vllm/torch_compile_cache/35745ab5a8
/home/mp/.cache/vllm/torch_compile_cache/39e38de6ea
/home/mp/.cache/vllm/torch_compile_cache/39f305425e
/home/mp/.cache/vllm/torch_compile_cache/3bb37fe4a8
/home/mp/.cache/vllm/torch_compile_cache/3c4e305f12
/home/mp/.cache/vllm/torch_compile_cache/3c6550008b
/home/mp/.cache/vllm/torch_compile_cache/3e2c800b7b
/home/mp/.cache/vllm/torch_compile_cache/3fafc45400
/home/mp/.cache/vllm/torch_compile_cache/3fc211650b
/home/mp/.cache/vllm/torch_compile_cache/4165db57c5
/home/mp/.cache/vllm/torch_compile_cache/419af37918
/home/mp/.cache/vllm/torch_compile_cache/41ab0e8e97
/home/mp/.cache/vllm/torch_compile_cache/42ab4909f8
/home/mp/.cache/vllm/torch_compile_cache/4748f5c69d
/home/mp/.cache/vllm/torch_compile_cache/474bf5d959
/home/mp/.cache/vllm/torch_compile_cache/47f0a28f34
/home/mp/.cache/vllm/torch_compile_cache/4826d51c4f
/home/mp/.cache/vllm/torch_compile_cache/4bd947bad1
/home/mp/.cache/vllm/torch_compile_cache/4c00d0c010
/home/mp/.cache/vllm/torch_compile_cache/4dcc58ac4e
/home/mp/.cache/vllm/torch_compile_cache/4fc83588d3
/home/mp/.cache/vllm/torch_compile_cache/510f68636c
/home/mp/.cache/vllm/torch_compile_cache/52ca55032f
/home/mp/.cache/vllm/torch_compile_cache/53047dc8ec
/home/mp/.cache/vllm/torch_compile_cache/5474878149
/home/mp/.cache/vllm/torch_compile_cache/56ecd7f7cf
/home/mp/.cache/vllm/torch_compile_cache/5ebfb04aa0
/home/mp/.cache/vllm/torch_compile_cache/60e338e8df
/home/mp/.cache/vllm/torch_compile_cache/61073645a2
/home/mp/.cache/vllm/torch_compile_cache/62ee263d78
/home/mp/.cache/vllm/torch_compile_cache/63efe2ab93
/home/mp/.cache/vllm/torch_compile_cache/653207e382
/home/mp/.cache/vllm/torch_compile_cache/6538fe21ba
/home/mp/.cache/vllm/torch_compile_cache/665f56aa31
/home/mp/.cache/vllm/torch_compile_cache/681c023aa2
/home/mp/.cache/vllm/torch_compile_cache/6961ed6789
/home/mp/.cache/vllm/torch_compile_cache/6e1d71304c
/home/mp/.cache/vllm/torch_compile_cache/706cd557e1
/home/mp/.cache/vllm/torch_compile_cache/72405de1b6
/home/mp/.cache/vllm/torch_compile_cache/73860a2471
/home/mp/.cache/vllm/torch_compile_cache/77070253c5
/home/mp/.cache/vllm/torch_compile_cache/7785208f65
/home/mp/.cache/vllm/torch_compile_cache/7836c6d756
/home/mp/.cache/vllm/torch_compile_cache/78b409005c
/home/mp/.cache/vllm/torch_compile_cache/7943675d0b
/home/mp/.cache/vllm/torch_compile_cache/7e076a30b6
/home/mp/.cache/vllm/torch_compile_cache/7e14077e74
/home/mp/.cache/vllm/torch_compile_cache/8021066cad
/home/mp/.cache/vllm/torch_compile_cache/81667ffc9f
/home/mp/.cache/vllm/torch_compile_cache/82d67794dc
/home/mp/.cache/vllm/torch_compile_cache/8449bb48cb
/home/mp/.cache/vllm/torch_compile_cache/85139ad015
/home/mp/.cache/vllm/torch_compile_cache/9139c06131
/home/mp/.cache/vllm/torch_compile_cache/927d3bc357
/home/mp/.cache/vllm/torch_compile_cache/93ee3c3dd9
/home/mp/.cache/vllm/torch_compile_cache/9cc96a9582
/home/mp/.cache/vllm/torch_compile_cache/9e4f10489d
/home/mp/.cache/vllm/torch_compile_cache/9e7a0aa062
/home/mp/.cache/vllm/torch_compile_cache/9f6e53bd0f
/home/mp/.cache/vllm/torch_compile_cache/a0ca5cfb1b
/home/mp/.cache/vllm/torch_compile_cache/a14176d671
/home/mp/.cache/vllm/torch_compile_cache/a21a739b79
/home/mp/.cache/vllm/torch_compile_cache/a22e3e836c
/home/mp/.cache/vllm/torch_compile_cache/a32ea330e5
/home/mp/.cache/vllm/torch_compile_cache/a5ac4d9bb0
/home/mp/.cache/vllm/torch_compile_cache/a683d962b8
/home/mp/.cache/vllm/torch_compile_cache/a6f69c2003
/home/mp/.cache/vllm/torch_compile_cache/a7c2185e96
/home/mp/.cache/vllm/torch_compile_cache/a82074bac1
/home/mp/.cache/vllm/torch_compile_cache/aad1fe4441
/home/mp/.cache/vllm/torch_compile_cache/ac5bc273a7
/home/mp/.cache/vllm/torch_compile_cache/aedab3daba
/home/mp/.cache/vllm/torch_compile_cache/af800de7bb
/home/mp/.cache/vllm/torch_compile_cache/b00622ad4c
/home/mp/.cache/vllm/torch_compile_cache/b16028c49d
/home/mp/.cache/vllm/torch_compile_cache/b2c59d3b6f
/home/mp/.cache/vllm/torch_compile_cache/b399602d3e
/home/mp/.cache/vllm/torch_compile_cache/b40ef6d7db
/home/mp/.cache/vllm/torch_compile_cache/b42e194964
/home/mp/.cache/vllm/torch_compile_cache/b449c8f368
/home/mp/.cache/vllm/torch_compile_cache/b5109f7a75
/home/mp/.cache/vllm/torch_compile_cache/b64c97d453
/home/mp/.cache/vllm/torch_compile_cache/bcc873df08
/home/mp/.cache/vllm/torch_compile_cache/bd37b6a746
/home/mp/.cache/vllm/torch_compile_cache/be00cbe5d6
/home/mp/.cache/vllm/torch_compile_cache/be6f808f26
/home/mp/.cache/vllm/torch_compile_cache/c09aa73fc3
/home/mp/.cache/vllm/torch_compile_cache/c271dd7f84
/home/mp/.cache/vllm/torch_compile_cache/c4874bb2ad
/home/mp/.cache/vllm/torch_compile_cache/c80489f9e3
/home/mp/.cache/vllm/torch_compile_cache/cb2c90c8d7
/home/mp/.cache/vllm/torch_compile_cache/ccded0d304
/home/mp/.cache/vllm/torch_compile_cache/cd59f1c20c
/home/mp/.cache/vllm/torch_compile_cache/d1a9f06641
/home/mp/.cache/vllm/torch_compile_cache/d29a0e5393
/home/mp/.cache/vllm/torch_compile_cache/d2ce8e0c36
/home/mp/.cache/vllm/torch_compile_cache/d300c26df6
/home/mp/.cache/vllm/torch_compile_cache/d494c9237f
/home/mp/.cache/vllm/torch_compile_cache/d74f1554d4
/home/mp/.cache/vllm/torch_compile_cache/da88f64e99
/home/mp/.cache/vllm/torch_compile_cache/dcb166b8af
/home/mp/.cache/vllm/torch_compile_cache/dfe95ea30f
/home/mp/.cache/vllm/torch_compile_cache/e3b29d44fe
/home/mp/.cache/vllm/torch_compile_cache/e56f6f22fb
/home/mp/.cache/vllm/torch_compile_cache/e6a43c3ad6
/home/mp/.cache/vllm/torch_compile_cache/e82acb307d
/home/mp/.cache/vllm/torch_compile_cache/ea2372db38
/home/mp/.cache/vllm/torch_compile_cache/eb0b506a87
/home/mp/.cache/vllm/torch_compile_cache/ec9466d6d2
/home/mp/.cache/vllm/torch_compile_cache/eee18cf42f
/home/mp/.cache/vllm/torch_compile_cache/f31af31c0c
/home/mp/.cache/vllm/torch_compile_cache/f3e3f24875
/home/mp/.cache/vllm/torch_compile_cache/f4ad1622ac
/home/mp/.cache/vllm/torch_compile_cache/f4d1d04ea2
/home/mp/.cache/vllm/torch_compile_cache/f55a9f8667
/home/mp/.cache/vllm/torch_compile_cache/f8a08bf1e9
/home/mp/.cache/vllm/torch_compile_cache/f920617cd1
/home/mp/.cache/vllm/torch_compile_cache/fbca17daeb
/home/mp/.cache/vllm/torch_compile_cache/torch_aot_compile

## 3. Arbeitsbäume gemergter/geschlossener PRs (git worktree remove, Zweig bleibt) — 12108 MiB
/home/mp/Projekte/vllm-research/1Cat-vLLM-lint-pr-main-test-repairs
/home/mp/Projekte/vllm-research/1Cat-vLLM-lint-pr-ple-row-gather-policy
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-blockpack
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-bmm
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-capcap
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-compilecache
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-devzero
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-directio
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-draftrows
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-dsv4buckets
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-evict
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-fa2sm75
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-fp8tune
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-p1
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-p7
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-pleadmit
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-plecascade
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-pptoken
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-qpn8
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-qpn8volta
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-s3
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-s3a
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-skinny
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-sparsepolicy
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-swa
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-testdt
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-tests
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-tmpeak

## 4. Mess-/Debug-Arbeitsbäume (git worktree remove, Zweig bleibt) — 29006 MiB
/home/mp/Projekte/vllm-research/1Cat-vLLM-885-base
/home/mp/Projekte/vllm-research/1Cat-vLLM-forkmain-noC
/home/mp/Projekte/vllm-research/1Cat-vLLM-mainchk
/home/mp/Projekte/vllm-research/1Cat-vLLM-oldprod
/home/mp/Projekte/vllm-research/1Cat-vLLM-pledisk
/home/mp/Projekte/vllm-research/1Cat-vLLM-pledisk-verify
/home/mp/Projekte/vllm-research/1Cat-vLLM-ppisend
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-pledisk-main
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-pledisk885
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-qsapage4
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-syncpp
/home/mp/Projekte/vllm-research/1Cat-vLLM-pr837
/home/mp/Projekte/vllm-research/1Cat-vLLM-qsa-dump
/home/mp/Projekte/vllm-research/1Cat-vLLM-scan
/home/mp/Projekte/vllm-research/1Cat-vLLM-syncpp

## 5. git worktree prune (6 Einträge, Verzeichnisse schon weg)

## 6. Lose Blobs in ~/.cache/huggingface/hub/blobs (kein Modell verweist darauf; c7 = opt-125m bleibt) — ~3070 MiB
/home/mp/.cache/huggingface/hub/blobs/53
/home/mp/.cache/huggingface/hub/blobs/79
/home/mp/.cache/huggingface/hub/blobs/88
/home/mp/.cache/huggingface/hub/blobs/89
/home/mp/.cache/huggingface/hub/blobs/9a
/home/mp/.cache/huggingface/hub/blobs/ca

## 7. v100-skinny/tools/bench_qsa_sparse_prefill.py (unversioniert, 11.10., überholt durch qsa_splitk_tile_bench.py 17.10. und #1236)

## 8. Optional: ~/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal (1479 MiB, in keinem llama-swap-Eintrag)
