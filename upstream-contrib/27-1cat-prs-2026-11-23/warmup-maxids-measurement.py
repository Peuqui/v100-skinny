import sys, torch
sys.path.insert(0, "tests/v1/worker")
from test_gpu_warmup_blocks import _glm53_warmup_runner
from vllm.v1.worker.gpu.warmup import warmup_kernels
torch.accelerator.synchronize = lambda: None
for k, n in [(3, 131), (4, 130), (5, 130), (7, 126)]:
    ex = []
    try:
        warmup_kernels(_glm53_warmup_runner(k, 10**6), ex.append, lambda *_: None)
    except AssertionError as e:
        print("assert", e)
    ids = [i for o in ex for r in o.scheduled_new_reqs for g in r.block_ids for i in g]
    ids += [i for o in ex for gs in o.scheduled_cached_reqs.new_block_ids if gs is not None for g in gs for i in g]
    print(f"k={k} pool={n} max_block_id={max(ids)} inside_pool={max(ids) < n}")
