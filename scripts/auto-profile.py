#!/usr/bin/env python3
"""Detect safe Linux Docker Compose backend; does not deploy or change Docker configuration."""

from __future__ import annotations

import argparse
from contextlib import suppress
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import cast


def detect_backend(
    *,
    nvidia_available: bool,
    render_node: bool,
    vulkan_healthy: bool,
    render_gid: int | None,
    cpu_threads: int,
) -> dict[str, object]:
    if nvidia_available:
        backend, compose = "cuda", "compose.cuda.yaml"
    elif render_node and vulkan_healthy:
        backend, compose = "vulkan", "compose.vulkan.yaml"
    else:
        backend, compose = "cpu", "compose.cpu.yaml"
    settings = {"TRANSCRIBE_BACKEND": backend}
    if backend == "vulkan" and render_gid is not None:
        settings["RENDER_GID"] = str(render_gid)
        settings["WHISPER_DEVICE"] = "cpu"  # faster-whisper/CTranslate2 has no AMD Vulkan path
        settings["WHISPER_COMPUTE_TYPE"] = "int8"
    elif backend == "cpu":
        settings["WHISPER_DEVICE"] = "cpu"
        settings["WHISPER_COMPUTE_TYPE"] = "int8"
    settings["WHISPER_CPU_THREADS"] = str(max(1, min(4, cpu_threads // 2)))
    return {
        "backend": backend,
        "compose": compose,
        "environment": settings,
        "commands": ["docker compose -f " + compose + " config", "docker compose -f " + compose + " up -d"],
        "explanation": [
            "Selection requires device access and a functioning host Vulkan loader when choosing Vulkan.",
            "GPU detection alone is NOT a performance benchmark or a guarantee of GPU operation in Docker.",
            "Check container /health, GPU utilization and one real transcription after startup.",
            "CPU-only is preferred to a broken GPU configuration; no mount or model files are changed.",
        ],
    }


def actual_probe() -> dict[str, object]:
    nvidia = False
    if binary := shutil.which("nvidia-smi"):
        with suppress(OSError, subprocess.TimeoutExpired):
            nvidia = subprocess.run([binary, "-L"], capture_output=True, timeout=5, check=False).returncode == 0
    render = sorted(Path("/dev/dri").glob("renderD*"))
    render = [r for r in render if os.access(r, os.R_OK | os.W_OK)]
    gid = render[0].stat().st_gid if render else None
    vulkan = False
    if render and (binary := shutil.which("vulkaninfo")):
        with suppress(OSError, subprocess.TimeoutExpired):
            vulkan = subprocess.run([binary, "--summary"], capture_output=True, timeout=12, check=False).returncode == 0
    return detect_backend(
        nvidia_available=nvidia,
        render_node=bool(render),
        vulkan_healthy=vulkan,
        render_gid=gid,
        cpu_threads=max(1, len(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", action="store_true", help="Print .env-ready recommended keys")
    args = parser.parse_args()
    result = actual_probe()
    if args.env:
        for k, v in cast("dict[str, str]", result["environment"]).items():
            print(f"{k}={v}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
