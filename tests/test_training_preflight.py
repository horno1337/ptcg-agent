"""Resource preflight checks. Run with ``python tests/test_training_preflight.py``."""

from __future__ import annotations

import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from tools.training_preflight import GIB, ResourceSnapshot, assess, gib, parse_meminfo


def test_meminfo_parser_uses_available_and_swap_free():
    available, swap = parse_meminfo(
        "MemTotal: 16000000 kB\n"
        "MemAvailable: 7000000 kB\n"
        "SwapTotal: 8000000 kB\n"
        "SwapFree: 3000000 kB\n"
    )
    assert available == 7_000_000 * 1024
    assert swap == 3_000_000 * 1024


def test_gib_accepts_argparse_strings():
    assert gib("6") == 6 * GIB
    assert gib("1.5") == int(1.5 * GIB)


def test_assess_fails_each_required_resource_independently():
    resources = ResourceSnapshot(
        available_memory_bytes=5 * GIB,
        free_swap_bytes=2 * GIB,
        gpu_free_bytes=(3 * GIB, 4 * GIB),
    )
    failures = assess(
        resources,
        min_available_bytes=6 * GIB,
        min_swap_free_bytes=4 * GIB,
        require_gpu=True,
        min_gpu_free_bytes=6 * GIB,
    )
    assert len(failures) == 3
    assert failures[0].startswith("available memory")
    assert failures[1].startswith("free swap")
    assert failures[2].startswith("selected GPU 0 free memory")


def test_assess_checks_selected_gpu_not_freest_gpu():
    resources = ResourceSnapshot(
        available_memory_bytes=8 * GIB,
        free_swap_bytes=5 * GIB,
        gpu_free_bytes=(2 * GIB, 12 * GIB),
    )
    failures = assess(
        resources,
        min_available_bytes=6 * GIB,
        min_swap_free_bytes=4 * GIB,
        require_gpu=True,
        min_gpu_free_bytes=6 * GIB,
        gpu_index=0,
    )
    assert failures == [
        "selected GPU 0 free memory 2.00 GiB is below 6.00 GiB"
    ]
    assert assess(
        resources,
        min_available_bytes=6 * GIB,
        min_swap_free_bytes=4 * GIB,
        require_gpu=True,
        min_gpu_free_bytes=6 * GIB,
        gpu_index=1,
    ) == []


def test_optional_gpu_does_not_block_cpu_training():
    resources = ResourceSnapshot(8 * GIB, 5 * GIB, ())
    assert assess(
        resources,
        min_available_bytes=6 * GIB,
        min_swap_free_bytes=4 * GIB,
        require_gpu=False,
        min_gpu_free_bytes=6 * GIB,
    ) == []


if __name__ == "__main__":
    test_meminfo_parser_uses_available_and_swap_free()
    test_gib_accepts_argparse_strings()
    test_assess_fails_each_required_resource_independently()
    test_assess_checks_selected_gpu_not_freest_gpu()
    test_optional_gpu_does_not_block_cpu_training()
    print("all training preflight tests passed")
