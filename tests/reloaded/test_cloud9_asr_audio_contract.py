"""Regression tests for immutable French audio benchmark fixture preparation.

Pure local synthetic audio, no model download and no GPU usage.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

MODULE = Path(__file__).resolve().parents[2] / "scripts/cloud9_asr_vulkan_ab.py"
spec = importlib.util.spec_from_file_location("cloud9_asr_vulkan_ab", MODULE)
assert spec is not None and spec.loader is not None
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.mark.parametrize("source_hz", [16000, 44100, 48000])
def test_prepares_mono_pcm16k_without_distorting_duration(tmp_path: Path, source_hz: int) -> None:
    signal = np.sin(2 * np.pi * 440.0 * np.arange(source_hz, dtype="float32") / source_hz) * 0.1
    wav = tmp_path / f"test_{source_hz}.wav"
    sf.write(wav, signal, source_hz)
    audio, original, target, duration = bench.load_audio_16k(wav)
    assert original == source_hz
    assert target == 16000
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert abs(duration - 1.0) < 0.005
    assert len(audio) == 16000
    assert np.isfinite(audio).all()
    assert float(np.max(np.abs(audio))) > 0.07


def test_stereo_fixture_rejected_instead_of_silently_mixing_channels(tmp_path: Path) -> None:
    wav = tmp_path / "stereo.wav"
    sf.write(wav, np.zeros((16000, 2), dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="Invalid audio shape"):
        bench.load_audio_16k(wav)
