"""Unit-Test der automatischen PLE-Platzierung (reine Rechnung, ohne GPU)."""
import os

GIB = 1024**3
FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main():
    from vllm.models.qwen4_exp.common.ple import (
        auto_ple_host_budget_bytes,
        available_host_bytes,
    )
    from vllm.models.qwen4_exp.amd.ple_layer import (
        _ple_host_budget_bytes,
        _ple_vram_reserve_bytes,
    )

    print("== Env-Semantik ==")
    for raw, want in (("", None), ("auto", None), ("AUTO", None), ("4", 4 * GIB),
                      ("2.5", int(2.5 * GIB)), ("0", 0)):
        os.environ["VLLM_QWEN4EXP_PLE_HOST_GIB"] = raw
        got = _ple_host_budget_bytes()
        check(f"{raw!r} -> {want}", got == want, f"got {got}")
    os.environ["VLLM_QWEN4EXP_PLE_HOST_GIB"] = "-1"
    try:
        _ple_host_budget_bytes()
        check("negative rejected", False)
    except ValueError:
        check("negative rejected", True)
    os.environ.pop("VLLM_QWEN4EXP_PLE_HOST_GIB")

    print("\n== Reserve ==")
    os.environ.pop("VLLM_QWEN4EXP_PLE_VRAM_RESERVE", None)
    check("default 6% of card", _ple_vram_reserve_bytes(47 * GIB) == int(47 * GIB * 0.06))
    os.environ["VLLM_QWEN4EXP_PLE_VRAM_RESERVE"] = "0.10"
    check("override honoured", _ple_vram_reserve_bytes(47 * GIB) == int(47 * GIB * 0.10))
    os.environ["VLLM_QWEN4EXP_PLE_VRAM_RESERVE"] = "1.5"
    try:
        _ple_vram_reserve_bytes(47 * GIB)
        check("out-of-range rejected", False)
    except ValueError:
        check("out-of-range rejected", True)
    os.environ.pop("VLLM_QWEN4EXP_PLE_VRAM_RESERVE")

    print("\n== Kaskade: Tabelle bleibt im VRAM, solange der Kontext passt ==")
    # Reale Groessen dieses Deployments: 47 GiB Karte, gmu 0.9, Tabelle 23.84 GiB,
    # uebrige Gewichte der Stufe rund 14.8 GiB bei Split 18/30.
    card, table, weights = 47 * GIB, int(23.84 * GIB), int(14.81 * GIB)
    reserve = _ple_vram_reserve_bytes(card)

    def budget(kv_gib, weights_bytes=weights):
        return auto_ple_host_budget_bytes(
            table_bytes=table,
            device_total_bytes=card,
            device_allocated_bytes=weights_bytes,
            gpu_memory_utilization=0.9,
            kv_cache_bytes=int(kv_gib * GIB),
            reserve_bytes=reserve,
        )

    check("tiny context spills nothing", budget(0.2) == 0, f"{budget(0.2)/GIB:.2f} GiB")
    b3 = budget(3.0)
    b6 = budget(6.0)
    check("spilling is monotone in the context", 0 <= b3 <= b6,
          f"{b3/GIB:.2f} -> {b6/GIB:.2f} GiB")
    check("spill never exceeds what the context claims",
          b6 - b3 <= int(3.0 * GIB) + 1, f"delta {(b6-b3)/GIB:.2f} GiB")
    check("6 GiB context starts spilling", 0 < b6 < table, f"{b6/GIB:.2f} GiB")
    b10 = budget(10.0)
    check("more context spills more", b10 > b6, f"{b10/GIB:.2f} vs {b6/GIB:.2f} GiB")
    huge = budget(40.0)
    check("impossible context clamps to table", huge == table, f"{huge/GIB:.2f} GiB")

    print("\n== Kaskade: mehr Layer auf der Stufe -> mehr Auslagerung ==")
    more_layers = weights + int(4 * 0.823 * GIB)     # Split 18/30 -> 22/26
    b_split18 = budget(5.0)
    b_split22 = budget(5.0, more_layers)
    check("heavier stage spills more", b_split22 > b_split18,
          f"{b_split22/GIB:.2f} vs {b_split18/GIB:.2f} GiB")

    print("\n== Host-Speicher ablesbar ==")
    avail = available_host_bytes()
    check("MemAvailable readable", avail is not None and avail > 0,
          f"{(avail or 0)/GIB:.1f} GiB")

    print(f"\n{'ALL PASS' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
