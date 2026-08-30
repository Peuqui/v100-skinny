"""Gitter-Experiment: Wo soll der MTP-Drafter sitzen?

Der Drafter lebt immer auf der LETZTEN PP-Stufe und macht pro Schritt k
sequenzielle Vollkontext-Durchlaeufe — er ist damit der latenzkritischste
Teil. Bisherige Konvention: schnellste Compute-Klasse zuerst, also RTX auf
Stufe 0 und der Drafter auf den V100. Seit dem FA2-Split-Fix ist offen, ob
das noch stimmt.

Usage: grid_stage_test.py <gpu-order> <partition> <spec-backend> <k> <tag>
  z.B. baseline:  0,2,1,4  38,26  FLASH_ATTN_V100  2  baseline
       reversed:  1,4,0,2  26,38  FLASH_ATTN       2  reversed
"""
import json, os, signal, subprocess, sys, time, urllib.request

GPUS, PART, SPEC_BE, K, TAG = sys.argv[1:6]
K = int(K)
# optional: max_num_batched_tokens, block_size
MBT = sys.argv[6] if len(sys.argv) > 6 else "2048"
BLK = sys.argv[7] if len(sys.argv) > 7 else "16"
PORT = 8129
VENV = "/home/mp/Projekte/v100-skinny/.venv-sm70-130"
MODEL = ("/home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/"
         "snapshots/554ebba9b5f1b79dc11246341960360e6ef05ef4")
NAME = "grid-test"
OUT = "/home/mp/Projekte/v100-skinny/hunt-results"
os.makedirs(OUT, exist_ok=True)

ENV = dict(os.environ)
ENV.update({
    "PATH": f"{VENV}/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": GPUS,
    "CUDA_HOME": "/home/mp/Projekte/v100-skinny/.cuda-nvcc-deb/usr/local/cuda-12.8",
    "TORCH_CUDA_ARCH_LIST": "7.0;7.5", "NCCL_P2P_DISABLE": "1",
    "VLLM_SM70_E5_CACHE": "0", "VLLM_SM70_NVFP4_TURBOMIND": "0",
    "VLLM_SM70_QUANT_BACKEND": "marlin", "VLLM_SKINNY_NVFP4": "1",
    "VLLM_SKINNY_QPN": "1", "VLLM_SKINNY_QPN2": "1",
    "VLLM_SKINNY_NVFP4_SRC": "/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu",
    "TORCHINDUCTOR_CACHE_DIR": "/home/mp/.cache/torchinductor",
    "VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS": "1",
    "VLLM_QWEN35_MTP_SHARE_IO_WEIGHTS": "0",
    "VLLM_PP_LAYER_PARTITION": PART, "HOME": "/home/mp",
})

spec = {"method": "mtp", "num_speculative_tokens": K,
        "draft_sample_method": "greedy", "use_local_argmax_reduction": True,
        "attention_backend": SPEC_BE}
args = [
    f"{VENV}/bin/python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL, "--served-model-name", NAME, "--trust-remote-code",
    "--dtype", "float16", "--disable-custom-all-reduce",
    "--tensor-parallel-size", "2", "--pipeline-parallel-size", "2",
    "--gpu-memory-utilization", "0.95", "--block-size", BLK,
    "--max-model-len", "262144", "--max-num-seqs", "4",
    "--max-num-batched-tokens", MBT, "--host", "127.0.0.1",
    "--port", str(PORT), "--language-model-only", "--async-scheduling",
    "--speculative-config", json.dumps(spec),
    "--compilation-config", json.dumps({"cudagraph_capture_sizes": [1, 2, 3, 4, 8]}),
]

log = open(f"{OUT}/grid_{TAG}.log", "w")
proc = subprocess.Popen(args, env=ENV, stdout=log, stderr=subprocess.STDOUT,
                        cwd="/home/mp")
print(f"[{TAG}] pid {proc.pid} | GPUs {GPUS} | Partition {PART} | "
      f"Drafter-Backend {SPEC_BE} | k={K} | MBT {MBT} | Block {BLK}")

def api(path, payload=None, timeout=1200):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def complete(prompt, max_tokens):
    t0 = time.time()
    r = api("/v1/completions", {"model": NAME, "prompt": prompt,
                                "max_tokens": max_tokens, "temperature": 0.0,
                                "ignore_eos": True})
    dt = time.time() - t0
    u = r["usage"]
    return u["prompt_tokens"], u["completion_tokens"], dt

try:
    for _ in range(400):
        if proc.poll() is not None:
            print(f"[{TAG}] SERVER TOT — letzte Zeilen:")
            log.flush(); os.system(f"tail -4 {OUT}/grid_{TAG}.log")
            sys.exit(1)
        try:
            api("/v1/models", timeout=5); break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError("boot timeout")
    print(f"[{TAG}] server up")

    # Kohaerenz zuerst — Tempo ohne korrekte Ausgabe ist wertlos
    checks = [("Die Hauptstadt von Frankreich ist Paris. Die Hauptstadt von "
               "Deutschland ist welche Stadt? Antworte in einem Satz.", "Berlin"),
              ("Rechne Schritt fuer Schritt und nenne am Ende das Ergebnis: "
               "Was ist 37 mal 43?", "1591"),
              ("Schreibe eine Python-Funktion, die einen String umdreht. "
               "Nur der Code, keine Erklaerung.", "[::-1]")]
    ok = 0
    for prompt, expect in checks:
        r = api("/v1/completions", {"model": NAME, "prompt": prompt,
                                    "max_tokens": 500, "temperature": 0.0})
        if expect in r["choices"][0]["text"]:
            ok += 1
    print(f"[{TAG}] KOHAERENZ: {ok}/{len(checks)}")

    short_p = ("Schreibe einen ausfuehrlichen Aufsatz ueber die Geschichte "
               "der Dampfmaschine und ihre Bedeutung.")
    complete(short_p, 20)
    _, n, dt = complete(short_p, 200)
    print(f"[{TAG}] SHORT  : {n/dt:.1f} tok/s")

    filler = "Die Industrialisierung veraenderte Europa grundlegend. " * 12 + "\n"
    long_p = filler * 260 + "\nFasse den Kern in einem Satz zusammen:"
    ptok, _, dt = complete(long_p, 1)
    print(f"[{TAG}] PREFILL: {ptok/dt:.0f} tok/s ({ptok} tok in {dt:.1f}s)")
    _, n, dt = complete(long_p, 200)
    print(f"[{TAG}] LONG   : {n/dt:.1f} tok/s")
finally:
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        proc.kill()
    print(f"[{TAG}] server stopped")
