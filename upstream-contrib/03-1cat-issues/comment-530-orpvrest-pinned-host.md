# Kommentar in #530 (orpvrest, PLE-Offload-Worker ausserhalb des Budgets) — GEPOSTET 2026-09-06 (Freigabe Peuqui)

A hint from a setup that hit the same table from the other side (2x RTX 8000 + 3x V100, 30 GB host RAM):

On SM70 the pinned-host path is the default as soon as VLLM_PLE_CPU_OFFLOAD is not set (_should_use_pinned_host_ple in qwen4_exp/nvidia/ple_layer.py). Each TP rank then keeps its shard of the FP8 table in pinned host memory and the GPU worker gathers the rows itself over UVA. There is no second process and no device allocation outside the worker, so nothing sits outside the profiler's budget. The price is host memory: every rank pins its own shard, for Flash Next that is 51 GB divided by TP, about 12.8 GB per rank and 51 GB for your TP4, and it has to be pinnable. VLLM_PLE_DISK_OFFLOAD is no escape from that: it only takes effect inside the offload process (_disk_offload = VLLM_PLE_DISK_OFFLOAD and is_offload_process()).

If the box has that much host RAM, dropping VLLM_PLE_CPU_OFFLOAD is a config-only test of whether the OOM goes away. #528 (splitting the pinned-host table between device and host by budget) buys nothing on your layout: with the KV pool at 3.8 GiB there is no room for table rows, it would degrade to the all-host path above.

Accounting for the offload child inside the memory profile stays the real fix for your configuration. We cannot test that path here (30 GB host RAM against the 51 GB table), which is why we left it alone; #479 lists what it needs under pipeline parallelism.
