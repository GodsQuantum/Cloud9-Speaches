from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_linux_x64_runtime_uses_cpu_only_pytorch_wheels() -> None:
    workflow = (ROOT / ".github/workflows/portable-desktop.yml").read_text(encoding="utf-8")
    assert 'if [ "$RUNNER_OS" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then' in workflow
    assert "--no-install-package torch --no-install-package torchaudio" in workflow
    assert "https://download.pytorch.org/whl/cpu" in workflow
    assert '"torch==2.8.0+cpu" "torchaudio==2.8.0+cpu"' in workflow
    assert "assert torch.version.cuda is None" in workflow
    assert "LINUX_CPU_PYTORCH_VERIFIED" in workflow


def test_linux_x64_rejects_oversized_archive() -> None:
    workflow = (ROOT / ".github/workflows/portable-desktop.yml").read_text(encoding="utf-8")
    assert "SPEACHES_LINUX_AMD64_SIZE_BUDGET_OK" in workflow
