"""E1/E2: Spec-Langkontext-Jagd — ein Boot, instrumentierte Messung.

Bootet das 27B auf waehlbarer Topologie mit MTP-Phasen-Profiler, misst
Kurz- und Lang-Decode und zieht waehrend des Lang-Decodes ein
py-spy-Profil des TP0-Workers.

Usage: spec_hunt.py <k> <gpus z.B. 0,2> [tp] [drafter-backend] [tag]
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

K = int(sys.argv[1])
GPUS = sys.argv[2]
TP = int(sys.argv[3]) if len(sys.argv) > 3 else 2
DRAFTER = sys.argv[4] if len(sys.argv) > 4 else "FLASH_ATTN"
TAG = sys.argv[5] if len(sys.argv) > 5 else f"k{K}"
PORT = 8127

VENV = "/home/mp/Projekte/v100-skinny/.venv-sm70-130"
PYSPY = f"{VENV}/bin/py-spy"
MODEL = ("/home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/"
         "snapshots/554ebba9b5f1b79dc11246341960360e6ef05ef4")
NAME = "hunt-27b"
OUT = "/home/mp/Projekte/v100-skinny/hunt-results"
os.makedirs(OUT, exist_ok=True)

ENV = dict(os.environ)
ENV.update({
    "PATH": f"{VENV}/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_HOME": "/home/mp/Projekte/v100-skinny/.cuda-nvcc-deb/usr/local/cuda-12.8",
    "TORCH_CUDA_ARCH_LIST": "7.5",
    "NCCL_P2P_DISABLE": "1",
    "VLLM_SM70_E5_CACHE": "0",
    "VLLM_SM70_NVFP4_TURBOMIND": "0",
    "VLLM_SM70_QUANT_BACKEND": "marlin",
    "VLLM_SKINNY_NVFP4": "1",
    "VLLM_SKINNY_QPN": "1",
    "VLLM_SKINNY_QPN2": "1",
    "VLLM_SKINNY_NVFP4_SRC": "/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu",
    "TORCHINDUCTOR_CACHE_DIR": "/home/mp/.cache/torchinductor",
    "CUDA_VISIBLE_DEVICES": GPUS,
    "VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS": "1" if K > 0 else "0",
    "HOME": "/home/mp",
    "VLLM_SM70_MTP_PROFILE": "1",
    "VLLM_SM70_MTP_PROFILE_INTERVAL": "8",
})

args = [
    f"{VENV}/bin/python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL, "--served-model-name", NAME,
    "--trust-remote-code", "--dtype", "float16",
    "--disable-custom-all-reduce", "--tensor-parallel-size", str(TP),
    "--pipeline-parallel-size", "1", "--gpu-memory-utilization", "0.95",
    "--block-size", "16", "--max-model-len", "65536",
    "--max-num-seqs", "4", "--max-num-batched-tokens", "2048",
    "--host", "127.0.0.1", "--port", str(PORT), "--language-model-only",
]
if K > 0:
    spec = {"method": "mtp", "num_speculative_tokens": K,
            "draft_sample_method": "greedy",
            "use_local_argmax_reduction": True,
            "attention_backend": DRAFTER}
    caps = sorted({1, 2, 4, K + 1, 8})
    args += ["--speculative-config", json.dumps(spec),
             "--compilation-config",
             json.dumps({"cudagraph_capture_sizes": caps})]

logpath = f"{OUT}/server_{TAG}.log"
log = open(logpath, "w")
proc = subprocess.Popen(args, env=ENV, stdout=log, stderr=subprocess.STDOUT,
                        cwd="/home/mp")
print(f"server pid {proc.pid}, tag={TAG}, gpus={GPUS}, k={K}, drafter={DRAFTER}")


def api(path, payload=None, timeout=1200):
    url = f"http://127.0.0.1:{PORT}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def complete(prompt, max_tokens):
    t0 = time.time()
    r = api("/v1/completions", {
        "model": NAME, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "ignore_eos": True})
    dt = time.time() - t0
    u = r["usage"]
    return u["prompt_tokens"], u["completion_tokens"], dt


def worker_pid():
    out = subprocess.run(["pgrep", "-f", "VLLM::Worker_TP0"],
                         capture_output=True, text=True).stdout.split()
    return int(out[-1]) if out else None


try:
    for _ in range(360):
        if proc.poll() is not None:
            print("SERVER DIED -- tail:")
            log.flush()
            os.system(f"tail -4 {logpath}")
            sys.exit(1)
        try:
            api("/v1/models", timeout=5)
            break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError("boot timeout")
    print("server up")

    short_p = ("Schreibe einen ausfuehrlichen Aufsatz ueber die Geschichte "
               "der Dampfmaschine und ihre Bedeutung.")
    complete(short_p, 20)  # warmup
    _, n, dt = complete(short_p, 200)
    print(f"SHORT decode: {n / dt:.1f} tok/s")

    filler = ("Die Industrialisierung veraenderte Europa grundlegend. " * 12
              + "\n")
    long_p = filler * 260 + "\nFasse den Kern in einem Satz zusammen:"
    ptok, _, dt = complete(long_p, 1)
    print(f"LONG prefill: {ptok} tok in {dt:.1f}s = {ptok / dt:.0f} tok/s")

    _, n, dt = complete(long_p, 200)
    print(f"LONG decode (cached): {n / dt:.1f} tok/s")

    # py-spy waehrend eines langen Decodes
    wpid = worker_pid()
    spy = None
    if wpid:
        spy = subprocess.Popen(
            [PYSPY, "record", "-p", str(wpid), "-d", "25", "-r", "200",
             "-f", "speedscope", "-o", f"{OUT}/pyspy_{TAG}.json"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"py-spy laeuft auf worker {wpid}")
    _, n, dt = complete(long_p, 600)
    print(f"LONG decode (600 tok, unter py-spy): {n / dt:.1f} tok/s")
    if spy:
        spy.wait(timeout=40)
        print(f"py-spy profil: {OUT}/pyspy_{TAG}.json")

    # MTP-Profil-Zeilen aus dem Serverlog ziehen
    log.flush()
    os.system(f"grep 'SM70 MTP proposer profile' {logpath} | tail -3")
finally:
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("server stopped")
