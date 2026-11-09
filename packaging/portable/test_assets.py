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



def test_model_aliases_independent_of_launch_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import speaches.model_aliases as aliases
    monkeypatch.chdir(tmp_path)
    aliases.load_model_id_aliases.cache_clear()
    assert aliases.MODEL_ID_ALIASES_PATH == ROOT / "model_aliases.json"
    assert aliases.resolve_model_id_alias("fr-quality") != "fr-quality"


def test_model_aliases_resolve_from_frozen_bundle(tmp_path: Path) -> None:
    import importlib
    import speaches.model_aliases as aliases

    (tmp_path / "model_aliases.json").write_text(
        '{"private-test-alias": "test/model"}', encoding="utf-8"
    )
    try:
        with (
            patch.object(sys, "frozen", new=True, create=True),
            patch.object(sys, "_MEIPASS", str(tmp_path), create=True),
        ):
            importlib.reload(aliases)
            assert aliases.MODEL_ID_ALIASES_PATH == tmp_path / "model_aliases.json"
            assert aliases.resolve_model_id_alias("private-test-alias") == "test/model"
    finally:
        importlib.reload(aliases)


def test_builder_embeds_acoustic_assets() -> None:
    build = (ROOT / "packaging/portable/build.py").read_text()
    assert 'f"{ROOT / \'model_aliases.json\'}:."' in build
    assert '"--collect-all",\n        "faster_whisper"' in build
    assert '"silero_encoder_v5.onnx"' in build
    assert '"silero_decoder_v5.onnx"' in build
