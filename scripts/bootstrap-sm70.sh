#!/usr/bin/env bash
# Bootstrap the SM70 NVFP4 serving stack from a clean checkout.
#
# Installs the pinned 1Cat-vLLM wheel into a fresh environment, deploys the
# fork patches over the installed package, and warms the skinny-kernel JIT
# build so the first served request does not pay for it.
#
# Nothing here is specific to our machines: every path is derived from the
# checkout or overridable by environment variable.
#
#   VLLM_WHEEL     OPTIONAL. Defaults to the pinned 1Cat-vLLM 1.2.2 release
#                  URL, whose SHA256 is pinned below and verified before
#                  installation. If you override it you MUST also set
#                  VLLM_WHEEL_SHA256 -- an unverified wheel is refused.
#   VLLM_WHEEL_SHA256  required alongside a custom VLLM_WHEEL
#                  1cat-vllm 1.2.2 — 1Cat AI's SM70/Volta fork of vLLM,
#                  Apache-2.0: https://github.com/1CatAI/1Cat-vLLM
#                  This project is a set of NVFP4/FP8 kernels and patches
#                  ON TOP of that engine; it is not an engine itself.
#   ENV_PREFIX     conda env prefix to create/use   (default: ./.venv-sm70)
#   PYTHON_VERSION default 3.12
#
# Usage:  bash scripts/bootstrap-sm70.sh          # pinned + digest-verified
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-$REPO_ROOT/.venv-sm70}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

die() { echo "ERROR: $*" >&2; exit 1; }
say() { echo "==> $*"; }

# ---------------------------------------------------------------- preflight
say "checking prerequisites"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found; need an NVIDIA driver"

CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | tr '\n' ' ')
echo "    compute capability: $CAPS"
case "$CAPS" in
  *7.0*) ;;
  *) die "no SM70 (compute 7.0 / Volta) GPU found — this stack targets V100.
       Found: $CAPS" ;;
esac

NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "    GPUs visible: $NGPU"
# The serving config is TP4 and the published measurements are 4x V100-16GB.
# Discover this now, not two steps into a build.
[ "$NGPU" -ge "${REQUIRE_GPUS:-4}" ] || die "this configuration needs ${REQUIRE_GPUS:-4} GPUs; $NGPU visible
       (set REQUIRE_GPUS=n only if you intend a different topology --
        the published numbers are TP4 on 4x V100-SXM2-16GB)"

# The skinny kernels are built by torch's JIT extension loader, which shells
# out to nvcc. It is frequently absent from PATH even where CUDA is installed.
NVCC="${NVCC:-$(command -v nvcc || true)}"
[ -n "$NVCC" ] || for c in /usr/local/cuda/bin/nvcc /usr/local/cuda-12.8/bin/nvcc; do
  [ -x "$c" ] && NVCC="$c" && break
done
[ -n "$NVCC" ] || die "nvcc not found. Install the CUDA toolkit or set NVCC=/path/to/nvcc"
echo "    nvcc: $NVCC ($("$NVCC" --version | tail -1))"
export PATH="$(dirname "$NVCC"):$PATH"

# The exact wheel every published number was measured on. Digest verified
# against the installed artifact on the reference machine.
PINNED_WHEEL_URL="https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.2.2/1cat_vllm-1.2.2-cp312-cp312-linux_x86_64.whl"
PINNED_WHEEL_SHA256="8a628983ad9d675559910372643220c418b307ddc7fd52ac65a7f5fbcb104bc6"
VLLM_WHEEL="${VLLM_WHEEL:-$PINNED_WHEEL_URL}"
VLLM_WHEEL_SHA256="${VLLM_WHEEL_SHA256:-}"
[ -n "$VLLM_WHEEL_SHA256" ] || \
  { [ "$VLLM_WHEEL" = "$PINNED_WHEEL_URL" ] && VLLM_WHEEL_SHA256="$PINNED_WHEEL_SHA256"; } || true

# The checkpoint is pinned too: a mutable ref silently changes the weights
# under the published results.
PINNED_MODEL_REPO="RadixArk/Qwen3.8-27B-NVFP4"
PINNED_MODEL_REVISION="554ebba9b5f1b79dc11246341960360e6ef05ef4"
echo "    checkpoint: $PINNED_MODEL_REPO @ $PINNED_MODEL_REVISION"
echo "      hf download $PINNED_MODEL_REPO --revision $PINNED_MODEL_REVISION"

# ------------------------------------------------------------------- python
if command -v conda >/dev/null; then
  say "creating environment at $ENV_PREFIX (python $PYTHON_VERSION)"
  conda create -y -p "$ENV_PREFIX" "python=$PYTHON_VERSION" >/dev/null
  PY="$ENV_PREFIX/bin/python"
else
  say "conda not found; using python -m venv at $ENV_PREFIX"
  python3 -m venv "$ENV_PREFIX"
  PY="$ENV_PREFIX/bin/python"
fi
"$PY" -m pip install --upgrade pip >/dev/null

say "installing pinned vLLM wheel"
# The README pins 1.2.2 and every published number was measured on it; 1Cat
# has since released 1.3.0, so an unchecked wheel silently changes the engine
# under the results. Verify the version, and the digest when one is supplied.
# Fetch a remote wheel to a local file FIRST, so the digest is checked against
# the bytes that get installed. Hashing only local paths let a URL install
# unverified, which is the weakest link in a "reproducible" bootstrap.
WHEEL_LOCAL="$VLLM_WHEEL"
case "$VLLM_WHEEL" in
  http://*|https://*)
    WHEEL_LOCAL="$(mktemp -d)/$(basename "${VLLM_WHEEL%%\?*}")"
    say "downloading wheel"
    curl -fL --retry 3 -o "$WHEEL_LOCAL" "$VLLM_WHEEL" || die "wheel download failed: $VLLM_WHEEL" ;;
esac
[ -f "$WHEEL_LOCAL" ] || die "wheel not found: $WHEEL_LOCAL"

# Fail CLOSED: no digest means no install, unless explicitly waived.
if [ -z "${VLLM_WHEEL_SHA256:-}" ]; then
  [ "${ALLOW_UNVERIFIED_WHEEL:-0}" = 1 ] || die \
    "no SHA256 for $VLLM_WHEEL
   Pin one:   VLLM_WHEEL_SHA256=<digest> ...
   Or waive:  ALLOW_UNVERIFIED_WHEEL=1 ...  (not reproducible)"
  echo "    WARNING: installing an unverified wheel" >&2
else
  got=$(sha256sum "$WHEEL_LOCAL" 2>/dev/null | cut -d" " -f1) \
    || got=$(shasum -a 256 "$WHEEL_LOCAL" | cut -d" " -f1)
  [ "$got" = "$VLLM_WHEEL_SHA256" ] || die \
    "wheel digest mismatch
   expected $VLLM_WHEEL_SHA256
   got      $got"
  echo "    wheel sha256 verified"
fi
"$PY" -m pip install "$WHEEL_LOCAL"

# Trust installed metadata, not the filename: a renamed wheel passes any
# filename check ever written.
INSTALLED=$("$PY" - <<'EOF'
import importlib.metadata as md
for n in ("1cat-vllm", "1cat_vllm"):
    try:
        print(md.distribution(n).version); break
    except Exception:
        pass
else:
    print("")
EOF
)
[ "$INSTALLED" = "1.2.2" ] || { [ "${ALLOW_WHEEL_MISMATCH:-0}" = 1 ] && \
    echo "    WARNING: installed 1cat-vllm $INSTALLED, not 1.2.2" >&2; } || die \
  "installed 1cat-vllm is '${INSTALLED:-<none>}', not 1.2.2
   Every published number was measured on 1.2.2. Override with
   ALLOW_WHEEL_MISMATCH=1 if you intend a different engine."
echo "    1cat-vllm $INSTALLED (from installed metadata)"

# tilelang must be pinned. The SM70 TileLang compile is fixed only in 0.1.10
# (upstream commit 0bb5cc4132); whatever pip resolves by default does not
# build on Volta, and the failure surfaces later inside GDN attention where it
# reads like a kernel bug rather than a dependency problem.
# apache-tvm-ffi MUST be pinned with it. tilelang 0.1.10 does not pin its own
# tvm-ffi, and current releases (0.1.13.post3 as of 2026-08) abort on import:
#   terminate called after throwing an instance of 'tvm::ffi::Error'
#     what(): TypeAttr `__ffi_repr__` is already registered for type index 132
# That is a hard crash, not a warning, and it only appears on a machine built
# today -- reference machines resolved 0.1.10 back when it was current. Found
# by the clean-clone run; without this pin the published command does not work.
say "pinning tilelang 0.1.10 + apache-tvm-ffi 0.1.10 (required for SM70)"
"$PY" -m pip install "tilelang==0.1.10" "apache-tvm-ffi==0.1.10"

# pip WILL print a red dependency-conflict block here: the 1cat-vllm 1.2.2 wheel
# declares tilelang==0.1.9 and apache-tvm-ffi==0.1.9. Overriding it is
# deliberate -- 0.1.9 does not build on Volta. Say so, because otherwise every
# user reasonably assumes the install broke.
echo "    NOTE: pip reported a dependency conflict against the wheel's declared"
echo "          tilelang/apache-tvm-ffi 0.1.9 pins. That is EXPECTED: 0.1.9 does"
echo "          not build on SM70. The versions installed above are correct."

# Verify the import actually works. This previously ran unchecked, so the
# tvm-ffi abort above scrolled past and the bootstrap carried on with a broken
# environment.
"$PY" -c "import tilelang; print('    tilelang', tilelang.__version__)" \
  || die "tilelang failed to import -- the environment is broken.
   This is usually an apache-tvm-ffi version mismatch; expected 0.1.10.
   Check: $PY -m pip show apache-tvm-ffi"

SP="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
[ -d "$SP/vllm" ] || die "vllm did not install into $SP"
say "site-packages: $SP"

# ------------------------------------------------------------- fork patches
# Copy tracked files over the installed package, keeping a .pre_bootstrap
# backup of whatever was there. Never hand-edit the installed file: the
# tracked copy in fork_patches/ is the reviewable source of truth.
say "deploying fork patches"
deploy() {  # $1 = tracked file, $2 = path under site-packages
  local src="$REPO_ROOT/fork_patches/$1" dst="$SP/$2"
  [ -f "$src" ] || die "missing tracked patch: $src"
  [ -f "$dst" ] || die "install path not found (wheel version mismatch?): $dst"
  [ -f "$dst.pre_bootstrap" ] || cp -p "$dst" "$dst.pre_bootstrap"
  cp -p "$src" "$dst"
  echo "    $1 -> $2"
}
deploy gdn_attn.py          vllm/v1/attention/backends/gdn_attn.py
deploy gpu_model_runner.py  vllm/v1/worker/gpu_model_runner.py
deploy marlin.py            vllm/model_executor/kernels/linear/nvfp4/marlin.py
deploy modelopt.py          vllm/model_executor/layers/quantization/modelopt.py
deploy torch_utils.py       vllm/utils/torch_utils.py
deploy attention.py         vllm/model_executor/layers/attention/attention.py
deploy custom_all_reduce.py vllm/distributed/device_communicators/custom_all_reduce.py
deploy qwen3_5_mtp.py       vllm/model_executor/models/qwen3_5_mtp.py
deploy qwen3_5.py           vllm/model_executor/models/qwen3_5.py
deploy cuda.py              vllm/platforms/cuda.py
deploy vllm_config.py       vllm/config/vllm.py
deploy kernel_config.py     vllm/config/kernel.py
deploy linear_init.py       vllm/model_executor/kernels/linear/__init__.py
deploy fp8.py               vllm/model_executor/layers/quantization/fp8.py
deploy compressed_tensors.py vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py
deploy deepseek_v4_nvidia_model.py vllm/models/deepseek_v4/nvidia/model.py
deploy deepseek_v4_mtp.py    vllm/models/deepseek_v4/nvidia/mtp.py
deploy nvfp4_moe_oracle.py  vllm/model_executor/layers/fused_moe/oracle/nvfp4.py
deploy nvfp4_emulation_moe.py vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py
deploy_new nvfp4_skinny_moe.py vllm/model_executor/layers/fused_moe/experts/nvfp4_skinny_moe.py
deploy sparse_attn_indexer.py vllm/model_executor/layers/sparse_attn_indexer.py
deploy mhc_tilelang.py      vllm/model_executor/kernels/mhc/tilelang.py
deploy deepseek_v4_attention.py vllm/models/deepseek_v4/attention.py
deploy dsv4_cache_utils.py vllm/models/deepseek_v4/common/ops/cache_utils.py
deploy dsv4_fused_compress_quant_cache.py vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py
deploy dsv4_fused_indexer_q.py vllm/models/deepseek_v4/common/ops/fused_indexer_q.py
deploy dsv4_dspark.py vllm/models/deepseek_v4/nvidia/dspark.py
deploy dsv4_amd_rocm.py vllm/models/deepseek_v4/amd/rocm.py
deploy spec_decode_dspark.py vllm/v1/spec_decode/dspark.py
deploy flashmla_sparse.py vllm/v1/attention/backends/mla/flashmla_sparse.py
deploy sm70_turbomind.py vllm/model_executor/layers/quantization/sm70_turbomind.py
deploy dsv4_sm70_gemv.py vllm/models/deepseek_v4/sm70/gemv.py
deploy breakable_cudagraph.py vllm/compilation/breakable_cudagraph.py
deploy multiproc_executor.py vllm/v1/executor/multiproc_executor.py
deploy gpu_worker.py vllm/v1/worker/gpu_worker.py
deploy parallel_state.py vllm/distributed/parallel_state.py
deploy dsv4_compressor.py   vllm/models/deepseek_v4/compressor.py
deploy rocm_aiter_mla_sparse.py vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
deploy import_utils.py      vllm/utils/import_utils.py
deploy sparse_swa.py        vllm/v1/attention/backends/mla/sparse_swa.py
deploy_new() {  # like deploy, but the target may not exist in the wheel yet
  local src="$REPO_ROOT/fork_patches/$1" dst="$SP/$2"
  [ -f "$src" ] || die "missing tracked patch: $src"
  cp -p "$src" "$dst"
  echo "    $1 -> $2 (new)"
}
deploy_new qwen_gdn_linear_attn_sm75.py vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn_sm75.py
deploy_new gdn_attn_sm75.py vllm/v1/attention/backends/gdn_attn_sm75.py
deploy_new qpn8_blk.py      vllm/model_executor/kernels/linear/scaled_mm/qpn8_blk.py

# --- qwen4_exp port (Qwen3.8-Flash-Next) + supporting core patches ---
deploy _custom_ops.py vllm/_custom_ops.py
deploy compilation.py vllm/config/compilation.py
deploy scheduler.py vllm/config/scheduler.py
deploy speculative.py vllm/config/speculative.py
deploy attention_layer_base.py vllm/model_executor/layers/attention_layer_base.py
deploy abstract.py vllm/model_executor/layers/mamba/abstract.py
deploy qwen_gdn_linear_attn.py vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
deploy mamba_utils.py vllm/model_executor/layers/mamba/mamba_utils.py
deploy flashinfer_utils.py vllm/model_executor/layers/quantization/utils/flashinfer_utils.py
deploy vocab_parallel_embedding.py vllm/model_executor/layers/vocab_parallel_embedding.py
deploy weight_utils.py vllm/model_executor/model_loader/weight_utils.py
deploy config.py vllm/model_executor/models/config.py
deploy interfaces.py vllm/model_executor/models/interfaces.py
deploy qwen3_next.py vllm/model_executor/models/qwen3_next.py
deploy registry.py vllm/model_executor/models/registry.py
deploy utils.py vllm/model_executor/models/utils.py
deploy interface.py vllm/platforms/interface.py
deploy transformers_utils_config.py vllm/transformers_utils/config.py
deploy transformers_configs_init.py vllm/transformers_utils/configs/__init__.py
deploy_new qwen4_exp.py vllm/transformers_utils/configs/qwen4_exp.py
deploy model_arch_config_convertor.py vllm/transformers_utils/model_arch_config_convertor.py
deploy short_conv_attn.py vllm/v1/attention/backends/short_conv_attn.py
deploy backends_utils.py vllm/v1/attention/backends/utils.py
deploy kv_cache_coordinator.py vllm/v1/core/kv_cache_coordinator.py
deploy kv_cache_utils.py vllm/v1/core/kv_cache_utils.py
deploy sched_scheduler.py vllm/v1/core/sched/scheduler.py
deploy single_type_kv_cache_manager.py vllm/v1/core/single_type_kv_cache_manager.py
deploy kv_cache_interface.py vllm/v1/kv_cache_interface.py
deploy llm_base_proposer.py vllm/v1/spec_decode/llm_base_proposer.py
deploy_new spec_decode_qwen4_exp.py vllm/v1/spec_decode/qwen4_exp.py
deploy block_table.py vllm/v1/worker/gpu/block_table.py
deploy model_runner.py vllm/v1/worker/gpu/model_runner.py
deploy worker_mamba_utils.py vllm/v1/worker/mamba_utils.py
deploy worker_utils.py vllm/v1/worker/utils.py
deploy tilelang_target.py   tilelang/utils/target.py
deploy triton_unified_attention.py vllm/v1/attention/ops/triton_unified_attention.py
deploy triton_attn.py vllm/v1/attention/backends/triton_attn.py
# FA2-sm75 enablement (2026-08-29): gates for the Turing FlashAttention
# backend; the kernel itself (_vllm_fa2_C.abi3.so, ~152 MB, too big for
# git) is built from Peuqui/flash-attention branch sm75-enablement and
# copied from its build tree if present.
deploy flash_attn.py vllm/v1/attention/backends/flash_attn.py
deploy flash_attn_interface.py vllm/vllm_flash_attn/flash_attn_interface.py
FA2_SM75_SO="$HOME/Projekte/vllm-research/flash-attention-sm75/build-sm75/_vllm_fa2_C.abi3.so"
if [ -f "$FA2_SM75_SO" ]; then
    say "deploying FA2-sm75 kernel from $FA2_SM75_SO"
    cp "$FA2_SM75_SO" "$SP/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so"
else
    say "WARNING: FA2-sm75 kernel not found ($FA2_SM75_SO) - sm75 falls back to TRITON_ATTN"
fi
# sm75 pair needs the UNMODIFIED upstream FLA triton ops under their
# upstream path (the fork's layers/fla/ops are modified and wrong on sm75).
say "deploying upstream flash_linear_attention tree (sm75 GDN pair)"
mkdir -p "$SP/vllm/third_party/flash_linear_attention"
cp -r "$REPO_ROOT/fork_patches/flash_linear_attention/." \
      "$SP/vllm/third_party/flash_linear_attention/"

say "deploying qwen4_exp model tree (Qwen3.8-Flash-Next)"
mkdir -p "$SP/vllm/models/qwen4_exp"
cp -r "$REPO_ROOT/fork_patches/qwen4_exp_models/." \
  "$SP/vllm/models/qwen4_exp/"
# sm70_native_round.py is deliberately NOT installed — experimental, inert.

# ------------------------------------------------------------------ kernels
KERNEL_SRC="$REPO_ROOT/kernels/skinny_kernels.cu"
[ -f "$KERNEL_SRC" ] || die "kernel source missing: $KERNEL_SRC"
say "warming the skinny-kernel JIT build (first build takes a few minutes)"
# Build through the DEPLOYED shim, not a hand-rolled ext.load(). The server
# loads name="skinny_nvfp4_v11" with -O3 --use_fast_math -lineinfo
# (fork_patches/marlin.py:150). torch.utils.cpp_extension keys its build
# directory on the name, and differing flags force a rebuild anyway, so
# building a differently-named extension here warms nothing and validates
# nothing -- the real nvcc build would then land inside the server's boot
# wait. Calling the shim also proves the fork-patch deploy above landed.
CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$NVCC")")}" \
VLLM_SKINNY_NVFP4_SRC="$KERNEL_SRC" TORCH_CUDA_ARCH_LIST=7.0 "$PY" - <<'PYEOF'
import sys
from vllm.model_executor.kernels.linear.nvfp4.marlin import _get_skinny_ext
mod = _get_skinny_ext()
if mod is None:
    sys.exit("skinny extension failed to build (see nvcc output above)")
missing = [f for f in ("gemm_qpn2", "gemm_qpn8", "gemm_qpn8_mt2",
                       "gemm_qpn8_blk", "gemm_qpn8_blk_mt2")
           if not hasattr(mod, f)]
if missing:
    sys.exit(f"kernel built but missing entry points: {missing}")
print("    kernels built and all entry points present")
PYEOF

cat <<EOF

Bootstrap complete.

  environment : $ENV_PREFIX
  kernels     : $KERNEL_SRC
  next        : bash scripts/serve-qwen38-native.sh <checkpoint-dir>

Set VLLM_SKINNY_NVFP4_SRC=$KERNEL_SRC in the serving environment
(serve-qwen38-native.sh does this for you).
EOF
