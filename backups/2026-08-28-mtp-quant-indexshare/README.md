Backup vor der IndexShare-Untersuchung (28.08.2026 abends).

vorher-venv/   Kopien der venv-Dateien (.venv-sm70-130), die fuer eine
               Instrumentierung des QSA-Indexers in Frage kommen. Stand:
               unveraendert, identisch zum 1.3.0-Deploy + fork_patches.
               Rueckspielen: Datei an dieselbe Stelle unter
               .venv-sm70-130/lib/python3.12/site-packages/vllm/ kopieren.
modellordner/  config.json und model.safetensors.index.json des
               transplantierten Modells
               /home/mp/models/Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTPQ.
               Alles andere dort sind Symlinks und regenerierbar.
CHECKSUMS.txt  sha256 der gesicherten Python-Dateien.

Referenzstand zu diesem Backup: Commit 00becbb, 49,2 / 67,2 tok/s bei k=4.
