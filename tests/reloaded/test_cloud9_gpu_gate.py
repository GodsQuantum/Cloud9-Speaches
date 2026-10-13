"""Offline contracts for Cloud9 host GPU gates, not a substitute for GPU tests."""

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
GATE = runpy.run_path(str(ROOT / "scripts/cloud9-host-gpu-preflight.py"))
CHECK = GATE["check"]


def clean_snapshot() -> dict:
    return {
        "gtt_bytes": 2 * 1024**3,
        "mem_available_kb": 32 * 1024**2,
        "io_full_avg10": 0.0,
        "memory_full_avg10": 0.0,
        "swap_pages_delta_200ms": 0,
        "gpu_tasks_d": [],
        "recent_gpu_errors": [],
    }


def test_clean_host_evidence_only_passes() -> None:
    assert CHECK(clean_snapshot()) == []


def test_current_cloud9_stuck_gpu_refuses_before_docker() -> None:
    bad = clean_snapshot()
    bad["gtt_bytes"] = 11427979264
    bad["gpu_tasks_d"] = [
        {"pid": 2625774, "wchan": "drm_suballoc_new", "comm": "llama-server"},
        {"pid": 2683546, "wchan": "ttm_bo_evict_first", "comm": "cat"},
    ]
    problems = CHECK(bad)
    assert any("GTT" in x for x in problems)
    assert any("blocked GPU tasks" in x for x in problems)


def test_driver_errors_or_missing_evidence_refuse() -> None:
    bad = clean_snapshot()
    bad["recent_gpu_errors"] = ["amdgpu_vm_validate failed"]
    assert CHECK(bad)
    del bad["recent_gpu_errors"]
    assert CHECK(bad)


def test_swap_pressure_and_host_ram_refuse() -> None:
    bad = clean_snapshot()
    bad["mem_available_kb"] = 5 * 1024**2
    bad["swap_pages_delta_200ms"] = 200
    assert len(CHECK(bad)) >= 2


def test_shell_launcher_gate_first_and_explicit_gpu_run() -> None:
    src = (ROOT / "scripts/cloud9-host-vulkan-ab.sh").read_text()
    gate = src.index('python3 -B "$ROOT/scripts/cloud9-host-gpu-preflight.py"')
    docker = src.index("docker run")
    pct = src.index("pct exec")
    assert gate < pct < docker
    assert 'mode="${1:---check}"' in src
    assert "--device /dev/dri/renderD128" in src
    assert "flock -n 8" in src
