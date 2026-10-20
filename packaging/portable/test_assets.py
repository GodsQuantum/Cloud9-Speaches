from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_static_assets_independent_of_working_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from speaches.portable_assets import realtime_console_dist

    monkeypatch.chdir(tmp_path)
    assert realtime_console_dist() == ROOT / "realtime-console" / "dist"
    assert (realtime_console_dist() / "index.html").is_file()


def test_static_assets_resolve_from_pyinstaller_bundle(tmp_path: Path) -> None:
    from speaches.portable_assets import realtime_console_dist

    with patch.object(sys, "frozen", new=True, create=True), patch.object(sys, "_MEIPASS", str(tmp_path), create=True):
        assert realtime_console_dist() == tmp_path / "realtime-console" / "dist"


def test_portable_builder_embeds_static_assets() -> None:
    build = (ROOT / "packaging/portable/build.py").read_text()
    assert '"--add-data"' in build
    assert "realtime-console/dist" in build
    assert "cwd=" in build
