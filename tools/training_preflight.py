"""Fail-closed resource preflight for long local training jobs.

The replay corpus is larger than host memory and earlier eager loads were run
while another GPU-heavy application was active.  Trainers can call this tool
before allocating a model or opening the corpus; a failed preflight is a clean
configuration stop, never a partially written checkpoint.

Examples::

    python tools/training_preflight.py
    python tools/training_preflight.py --require-gpu --min-gpu-free-gib 6
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import subprocess
from typing import Iterable


GIB = 1024 ** 3


@dataclasses.dataclass(frozen=True)
class ResourceSnapshot:
    available_memory_bytes: int
    free_swap_bytes: int
    gpu_free_bytes: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "available_memory_gib": self.available_memory_bytes / GIB,
            "free_swap_gib": self.free_swap_bytes / GIB,
            "gpu_free_gib": [value / GIB for value in self.gpu_free_bytes],
        }


def parse_meminfo(text: str) -> tuple[int, int]:
    """Return Linux MemAvailable and SwapFree in bytes."""
    values: dict[str, int] = {}
    for raw_line in text.splitlines():
        key, separator, rest = raw_line.partition(":")
        if not separator:
            continue
        fields = rest.strip().split()
        if not fields:
            continue
        try:
            number = int(fields[0])
        except ValueError:
            continue
        unit = fields[1].lower() if len(fields) > 1 else "bytes"
        multiplier = 1024 if unit == "kb" else 1
        values[key] = number * multiplier
    missing = [key for key in ("MemAvailable", "SwapFree") if key not in values]
    if missing:
        raise ValueError(f"/proc/meminfo is missing {', '.join(missing)}")
    return values["MemAvailable"], values["SwapFree"]


def _gpu_free_bytes() -> tuple[int, ...]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ()
    values: list[int] = []
    for line in completed.stdout.splitlines():
        try:
            mib = int(line.strip())
        except ValueError:
            continue
        values.append(mib * 1024 ** 2)
    return tuple(values)


def snapshot(meminfo_path: str = "/proc/meminfo") -> ResourceSnapshot:
    with open(meminfo_path, encoding="utf-8") as handle:
        available, swap = parse_meminfo(handle.read())
    return ResourceSnapshot(available, swap, _gpu_free_bytes())


def assess(
    resources: ResourceSnapshot,
    *,
    min_available_bytes: int,
    min_swap_free_bytes: int,
    require_gpu: bool,
    min_gpu_free_bytes: int,
    gpu_index: int = 0,
) -> list[str]:
    failures: list[str] = []
    if resources.available_memory_bytes < min_available_bytes:
        failures.append(
            "available memory "
            f"{resources.available_memory_bytes / GIB:.2f} GiB is below "
            f"{min_available_bytes / GIB:.2f} GiB"
        )
    if resources.free_swap_bytes < min_swap_free_bytes:
        failures.append(
            f"free swap {resources.free_swap_bytes / GIB:.2f} GiB is below "
            f"{min_swap_free_bytes / GIB:.2f} GiB"
        )
    if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < 0:
        failures.append("GPU index must be a non-negative integer")
    elif require_gpu and not resources.gpu_free_bytes:
        failures.append("no NVIDIA GPU memory reading is available")
    elif require_gpu and gpu_index >= len(resources.gpu_free_bytes):
        failures.append(
            f"selected GPU index {gpu_index} is unavailable; "
            f"only {len(resources.gpu_free_bytes)} GPU reading(s) exist"
        )
    elif require_gpu and resources.gpu_free_bytes[gpu_index] < min_gpu_free_bytes:
        failures.append(
            f"selected GPU {gpu_index} free memory "
            f"{resources.gpu_free_bytes[gpu_index] / GIB:.2f} "
            f"GiB is below {min_gpu_free_bytes / GIB:.2f} GiB"
        )
    return failures


def gib(value: float | str) -> int:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "resource thresholds must be finite numbers"
        ) from error
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("resource thresholds must be non-negative")
    return int(parsed * GIB)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-available-gib", type=gib, default=gib(6.0))
    parser.add_argument("--min-swap-free-gib", type=gib, default=gib(4.0))
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--min-gpu-free-gib", type=gib, default=gib(6.0))
    parser.add_argument(
        "--gpu-index", type=int, default=0,
        help="physical nvidia-smi GPU index whose free memory must pass",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    resources = snapshot()
    failures = assess(
        resources,
        min_available_bytes=args.min_available_gib,
        min_swap_free_bytes=args.min_swap_free_gib,
        require_gpu=args.require_gpu,
        min_gpu_free_bytes=args.min_gpu_free_gib,
        gpu_index=args.gpu_index,
    )
    payload = {
        "schema": "ptcg-training-preflight-v1",
        "ok": not failures,
        "resources": resources.as_dict(),
        "thresholds": {
            "min_available_gib": args.min_available_gib / GIB,
            "min_swap_free_gib": args.min_swap_free_gib / GIB,
            "require_gpu": args.require_gpu,
            "min_gpu_free_gib": args.min_gpu_free_gib / GIB,
            "gpu_index": args.gpu_index,
        },
        "failures": failures,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
