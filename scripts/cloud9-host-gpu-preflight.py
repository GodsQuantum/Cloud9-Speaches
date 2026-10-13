#!/usr/bin/env python3
"""Read-only Proxmox HOST gate for Cloud9 GPU benchmarks. No driver actions."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

GTT = Path("/sys/class/drm/card0/device/mem_info_gtt_used")
GPU_WAIT = re.compile(r"amdgpu|drm_|ttm_|drm_suballoc", re.IGNORECASE)
GPU_COMM = re.compile(r"llama-server|strata|vulkan|rocm|hip", re.IGNORECASE)
GPU_ERROR = re.compile(
    r"amdgpu_vm_validate.+failed|Not enough memory for command submission"
    r"|GPU reset|ring .+ timeout|amdgpu.+GPU fault|device lost",
    re.IGNORECASE,
)


def read(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def mem_available_kb() -> int:
    match = re.search(r"^MemAvailable:\s+(\d+)\s+kB", read("/proc/meminfo"), re.MULTILINE)
    if not match:
        raise ValueError("missing MemAvailable")
    return int(match.group(1))


def psi_full(path: str) -> float:
    line = next(x for x in read(path).splitlines() if x.startswith("full "))
    return float(re.search(r"\bavg10=([\d.]+)", line).group(1))


def swap_io_pages() -> int:
    values = {
        p[0]: int(p[1]) for line in read("/proc/vmstat").splitlines() if len(p := line.split()) == 2 and p[1].isdigit()
    }
    return values["pswpin"] + values["pswpout"]


def live() -> dict[str, Any]:
    before = swap_io_pages()
    time.sleep(0.2)
    delta = max(0, swap_io_pages() - before)
    proc = subprocess.run(
        ["/usr/bin/ps", "-e", "-o", "pid=,stat=,wchan:32=,comm="], capture_output=True, text=True, check=True, timeout=8
    )
    tasks = []
    for line in proc.stdout.splitlines():
        fields = line.split(maxsplit=3)
        if len(fields) != 4:
            continue
        pid, state, wait, comm = fields
        if state.startswith("D") and (GPU_WAIT.search(wait) or GPU_COMM.search(comm)):
            tasks.append({"pid": int(pid), "wchan": wait, "comm": comm})
    logs = subprocess.run(
        ["/usr/bin/journalctl", "-k", "-b", "--since", "-3 minutes", "--no-pager", "-o", "cat"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return {
        "gtt_bytes": int(read(GTT).strip()),
        "mem_available_kb": mem_available_kb(),
        "io_full_avg10": psi_full("/proc/pressure/io"),
        "memory_full_avg10": psi_full("/proc/pressure/memory"),
        "swap_pages_delta_200ms": delta,
        "gpu_tasks_d": tasks,
        "recent_gpu_errors": [x[:240] for x in logs.stdout.splitlines() if GPU_ERROR.search(x)][-12:],
    }


def check(data: dict[str, Any]) -> list[str]:
    """Conservative host defaults: refuse rather than preempt any other workload."""
    required = (
        "gtt_bytes",
        "mem_available_kb",
        "io_full_avg10",
        "memory_full_avg10",
        "swap_pages_delta_200ms",
        "gpu_tasks_d",
        "recent_gpu_errors",
    )
    for key in required:
        if key not in data:
            return [f"incomplete evidence: {key}"]
    try:
        gtt = int(data["gtt_bytes"])
        mem = int(data["mem_available_kb"])
        io = float(data["io_full_avg10"])
        pressure = float(data["memory_full_avg10"])
        swap = int(data["swap_pages_delta_200ms"])
        tasks, errors = data["gpu_tasks_d"], data["recent_gpu_errors"]
        if not isinstance(tasks, list) or not isinstance(errors, list):
            raise TypeError("invalid task/error arrays")
        if min(gtt, mem, io, pressure, swap) < 0:
            raise ValueError("negative resource counter")
    except (TypeError, ValueError) as exc:
        return [f"invalid evidence: {exc}"]
    bad = []
    if gtt > 8192 * 1024**2:
        bad.append(f"GTT {gtt / 1024**3:.2f} GiB > 8 GiB idle limit")
    if mem < 16384 * 1024:
        bad.append(f"available RAM {mem / 1024:.0f} MiB < 16 GiB")
    if io > 5.0:
        bad.append(f"I/O full PSI {io} > 5")
    if pressure > 2.0:
        bad.append(f"memory full PSI {pressure} > 2")
    if swap > 64:
        bad.append(f"swap I/O {swap} pages / 200ms > 64")
    if tasks:
        bad.append(f"blocked GPU tasks in host: {tasks}")
    if errors:
        bad.append(f"GPU errors within last 3 minutes: {errors[-2:]}")
    return bad


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--fixture-stdin", action="store_true", help="offline unit-test input only; never used by host runner"
    )
    args = cli.parse_args()
    try:
        snapshot = json.load(sys.stdin) if args.fixture_stdin else live()
        problems = check(snapshot)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        problems = [f"cannot verify GPU safety: {exc}"]
    print(json.dumps({"status": "REFUSE" if problems else "PASS", "problems": problems}, ensure_ascii=False))
    return 4 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
