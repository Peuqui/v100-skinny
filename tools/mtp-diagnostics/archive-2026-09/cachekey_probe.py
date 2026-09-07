"""Beweis: aendert ein Kernel-Schalter den Compile-Cache-Schluessel?"""
import hashlib, json, os, sys
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    os.environ[k] = v
import vllm.envs as envs
f = envs.compile_factors()
blob = json.dumps({k: str(v) for k, v in sorted(f.items())}, ensure_ascii=False)
print("factors:", len(f), "env_hash:", hashlib.sha256(blob.encode()).hexdigest()[:16])
for name in ("VLLM_SKINNY_QPN", "VLLM_SKINNY_NVFP4", "VLLM_SM70_E5_CACHE",
             "VLLM_SM70_QPN8", "VLLM_SM70_NVFP4_TURBOMIND", "VLLM_SM70_QUANT_BACKEND"):
    print(f"  {name:32s} im Schluessel: {name in f}")
