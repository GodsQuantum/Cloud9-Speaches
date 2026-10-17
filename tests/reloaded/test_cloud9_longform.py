from __future__ import annotations

from itertools import pairwise
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from speaches.executors.cloud9_longform import ChunkProfile, chunk_windows, overlap_drop, profile_for_arch, run_longform


def test_profiles_choose_safe_arch_specific_lengths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUD9_LONGFORM_POLICY", raising=False)
    assert profile_for_arch("cohere_asr", 105) == ChunkProfile(34.0, 1.0)
    assert profile_for_arch("canary", 105) == ChunkProfile(30.0, 0.0)
    assert profile_for_arch("qwen3_asr", 105) == ChunkProfile(30.0, 0.0)
    assert profile_for_arch("whisper", 105) is None
    assert profile_for_arch("cohere_asr", 25) is None
    monkeypatch.setenv("CLOUD9_LONGFORM_POLICY", "all")
    assert profile_for_arch("whisper", 105) == ChunkProfile(30.0, 0.0)
    monkeypatch.setenv("CLOUD9_LONGFORM_POLICY", "off")
    assert profile_for_arch("cohere_asr", 105) is None


@pytest.mark.parametrize(
    ("duration", "profile"),
    [
        (105, ChunkProfile(34, 1)),
        (105, ChunkProfile(30, 0)),
        (3600, ChunkProfile(30, 1)),
        (25, ChunkProfile(30, 0)),
    ],
)
def test_windows_cover_every_sample_without_gaps(duration: int, profile: ChunkProfile) -> None:
    # Sparse float32 data can represent an hour without model allocations.
    sr = 160
    audio = np.zeros(duration * sr, dtype=np.float32)
    windows = chunk_windows(audio, sr, profile)
    assert windows[0][0] == 0
    assert windows[-1][1] == len(audio)
    for a, b in windows:
        assert 0 <= a < b <= len(audio)
    for (_, prev_end), (start, _) in pairwise(windows):
        assert start <= prev_end
    assert len(windows) < duration + 2


def test_overlap_words_preserves_punctuation_and_returns_confirmed_duplicate_size() -> None:
    previous = "Il a envoyé des messages sur les Noirs et les Arabes."
    new = "les Arabes, ça aurait fait deux sets de stand-up."
    assert overlap_drop(previous, new) == 2
    assert overlap_drop("Vive la France", "aujourd'hui que oui") == 0


def test_run_longform_native_segments_are_offset() -> None:
    sample_rate = 16000
    samples = np.full(sample_rate * 4, 0.02, dtype=np.float32)

    class Fake:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, _audio: np.ndarray, **_kw: Any) -> SimpleNamespace:
            self.calls += 1
            return SimpleNamespace(
                text="Bonjour le monde",
                language="fr",
                segments=[SimpleNamespace(text="Bonjour le monde", t0_ms=100, t1_ms=300)],
                words=[SimpleNamespace(text="Bonjour", t0_ms=100, t1_ms=200)],
            )

    result = run_longform(
        Fake(), samples, sample_rate=sample_rate, language="fr", timestamps="word", profile=ChunkProfile(2, 0, 0)
    )
    assert result.text == "Bonjour le monde Bonjour le monde"
    assert [(s.t0_ms, s.t1_ms) for s in result.segments] == [(100, 300), (2100, 2300)]
    assert [(w.t0_ms, w.t1_ms) for w in result.words] == [(100, 200), (2100, 2200)]


def test_run_longform_rejects_missing_speech() -> None:
    samples = np.ones(16000 * 3, dtype=np.float32) * 0.1

    class Fake:
        def run(self, *_args: Any, **_kw: Any) -> SimpleNamespace:
            return SimpleNamespace(text="", segments=(), words=())

    with pytest.raises(RuntimeError, match="omitted speech"):
        run_longform(Fake(), samples, sample_rate=16000, language="fr", timestamps="none", profile=ChunkProfile(2, 0))


def test_timestamp_none_segment_zero_duration_never_becomes_media_time_zero() -> None:
    class DecoderWithoutNativeTimestamps:
        def run(self, _audio: np.ndarray, **_kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                text="Le texte a été prononcé",
                segments=[SimpleNamespace(text="Le texte a été prononcé", t0_ms=0, t1_ms=0)],
                words=[],
            )

    audio = np.ones(16000 * 4, dtype=np.float32) * 0.02
    result = run_longform(
        DecoderWithoutNativeTimestamps(),
        audio,
        sample_rate=16000,
        language="fr",
        timestamps="none",
        profile=ChunkProfile(2, 0, 0),
    )
    assert [(r.t0_ms, r.t1_ms) for r in result.segments] == [(0, 2000), (2000, 4000)]
