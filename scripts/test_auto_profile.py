"""Hardware detector tests use synthetic devices to avoid requiring a GPU."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("auto_profile", Path(__file__).with_name("auto-profile.py"))
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestBackendChoice(unittest.TestCase):
    def test_cpu_safe(self) -> None:
        p = mod.detect_backend(
            nvidia_available=False, render_node=False, vulkan_healthy=False, render_gid=None, cpu_threads=4
        )
        assert p["backend"] == "cpu"

    def test_vulkan_requires_driver(self) -> None:
        p = mod.detect_backend(
            nvidia_available=False, render_node=True, vulkan_healthy=False, render_gid=105, cpu_threads=16
        )
        assert p["backend"] == "cpu"

    def test_vulkan_render_gid(self) -> None:
        p = mod.detect_backend(
            nvidia_available=False, render_node=True, vulkan_healthy=True, render_gid=105, cpu_threads=16
        )
        assert p["environment"]["RENDER_GID"] == "105"
        assert p["environment"]["WHISPER_DEVICE"] == "cpu"

    def test_cuda_selects_nvidia(self) -> None:
        p = mod.detect_backend(
            nvidia_available=True, render_node=False, vulkan_healthy=False, render_gid=None, cpu_threads=8
        )
        assert p["compose"] == "compose.cuda.yaml"


if __name__ == "__main__":
    unittest.main()
