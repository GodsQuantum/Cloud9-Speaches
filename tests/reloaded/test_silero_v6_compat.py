from pathlib import Path

import pytest

from speaches.executors import silero_vad_v5 as vad


def test_silero_v6_model_file_is_preferred(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_path = tmp_path / "silero_vad_v6.onnx"
    model_path.write_bytes(b"x")
    monkeypatch.setattr(vad, "get_assets_path", lambda: str(tmp_path))

    files = vad.silero_vad_model_registry.get_model_files(vad.MODEL_ID)

    assert files.model == model_path
