# Copyright (c) 2026 GodsQuantum
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from build import find_nsis_compiler
import pytest

if TYPE_CHECKING:
    from pathlib import Path


def test_require_nsis_uses_preinstalled_compiler(tmp_path: Path) -> None:
    compiler = tmp_path / "makensis.exe"
    compiler.write_bytes(b"real test marker")
    with patch("build.shutil.which", return_value=str(compiler)):
        assert find_nsis_compiler() == str(compiler)


def test_require_nsis_fails_without_preinstalled_compiler() -> None:
    with (
        patch("build.shutil.which", return_value=None),
        patch("build.Path.is_file", return_value=False),
        pytest.raises(RuntimeError, match="NSIS compiler"),
    ):
        find_nsis_compiler()
