#!/usr/bin/env bash
# Mini-spezifischer Bootstrap-Treiber (kein System-Toolkit nötig):
# CUDA-12.8-Compiler aus entpackten NVIDIA-debs (.cuda-nvcc-deb/, nvcc+nvvm+crt,
# via dpkg -x — nichts installiert) + Runtime-Header/-Libs aus dem pip-Paket
# nvidia-cuda-runtime-cu12, dann das offizielle bootstrap-sm70.sh.
set -euo pipefail
cd /home/mp/Projekte/v100-skinny

echo "== 1/3 venv + pip-Runtime"
[ -x .venv-sm70/bin/python ] || python3.12 -m venv .venv-sm70
.venv-sm70/bin/pip install -q --upgrade pip
.venv-sm70/bin/pip install -q "nvidia-cuda-runtime-cu12==12.8.*"

echo "== 2/3 CUDA_HOME komplettieren (Runtime-Header/-Libs in den deb-Baum linken)"
SP=$(.venv-sm70/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
CH=/home/mp/Projekte/v100-skinny/.cuda-nvcc-deb/usr/local/cuda-12.8
[ -x "$CH/bin/nvcc" ] || { echo "FEHLER: $CH/bin/nvcc fehlt (debs entpackt?)"; exit 1; }
ln -sfn "$SP/nvidia/cuda_runtime/include"/* "$CH/include/"
mkdir -p "$CH/lib64"
ln -sfn "$SP/nvidia/cuda_runtime/lib"/*.so* "$CH/lib64/" 2>/dev/null || true
"$CH/bin/nvcc" --version | tail -1

echo "== 3/3 offizielles bootstrap-sm70.sh"
NVCC="$CH/bin/nvcc" CUDA_HOME="$CH" REQUIRE_GPUS=3 \
  bash scripts/bootstrap-sm70.sh
echo "BOOTSTRAP-TREIBER FERTIG"
