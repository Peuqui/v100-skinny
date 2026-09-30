Title: [Bug][Test] Two tests fail on main: stale QSA E4M3 expectation, PLE offload config with one visible GPU

Two tests fail on current main (d3046986), on a Tesla V100 and on a Quadro RTX 8000 alike. No open PR fixes either: a search for the test names finds only my own #646 and #717, which list them as already failing on main.

**1. `tests/models/qwen4_exp/test_weight_loading.py::test_qsa_e4m3_loader_requires_all_24_scales`**

```
Failed: DID NOT RAISE ValueError
```

Since a286ed5b ("Serve an uncalibrated E4M3 QSA cache instead of refusing") `_validate_qsa_e4m3_scale_load` only warns about missing K/V scales and raises only with `VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES=1`. The test still expects the refusal without setting that switch, so it fails on every machine. Both paths are covered since 30ed6ebe by `tests/compile/test_qwen4exp_qsa_e4m3_gate.py` (`test_strict_mode_keeps_the_hard_failure`, `test_uncalibrated_checkpoint_degrades_to_a_warning`), so the old test can either set the switch (`monkeypatch.setattr(envs, "VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES", True)`, checked: it passes then) or go as a duplicate.

**2. `tests/v1/worker/test_ple_offload_worker.py::test_offload_distributed_sets_config_only_for_model_parallel`**

Fails when the test, or its whole file, runs with exactly one visible CUDA device; with two it passes:

```
CUDA_VISIBLE_DEVICES=4    pytest tests/v1/worker/test_ple_offload_worker.py  -> 1 failed, 28 passed, 1 skipped
CUDA_VISIBLE_DEVICES=1,3  pytest <this test>                                 -> 1 passed
```

In a larger run on one device it can pass, depending on which tests ran before it in the same process.

```
vllm/config/vllm.py:1800: in __post_init__
vllm/config/vllm.py:326: in _any_participating_device_is_capability
vllm/platforms/cuda.py:700: in _nvml_device_capability
vllm/platforms/cuda.py:661: in device_id_to_physical_device_id
E   IndexError: list index out of range
```

The test sets `VLLM_DP_RANK_LOCAL=1`, as a GPU worker of DP rank 1 would pass it down. `_init_offload_distributed` builds `VllmConfig()` and forces DP1 only afterwards, but `__post_init__` already derives the participating devices from the inherited DP rank (`_participating_cuda_device_ids`, df856016) and asks for device 1, which does not exist when one device is visible. Outside the test this would hit a PLE offload process of a DP rank that sees only its own GPU. Building the offload config with DP1 before `__post_init__` runs (or clearing the inherited DP variables for that construction) would avoid it.

Found while running our fork's test suite; happy to send a PR for either if that helps.

---
Status: gesendet 2026-09-30 als https://github.com/1CatAI/1Cat-vLLM/issues/739
