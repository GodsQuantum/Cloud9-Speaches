"""Regression: keep verified ASR window boundaries after forced CTC alignment."""
from types import SimpleNamespace

import pytest

from speaches.executors import transcribe_cpp


def request(duration: float) -> SimpleNamespace:
    return SimpleNamespace(
        response_format="verbose_json",
        timestamp_granularities=["word"],
        audio=SimpleNamespace(duration=duration),
        language="fr",
        alignment_model=None,
    )


def test_preserve_real_segments_when_native_word_timings_are_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUD9_WORD_ALIGNMENT_STRATEGY", "forced")
    monkeypatch.setenv("CLOUD9_WORD_ALIGNMENT_URL", "internal")
    segments = tuple(
        SimpleNamespace(text=t, t0_ms=a, t1_ms=b)
        for t, a, b in [("bonjour", 0, 30000), ("le", 30000, 60000), ("monde", 60000, 85000)]
    )
    response = SimpleNamespace(text="bonjour le monde", language="fr", segments=segments, words=())
    aligned = [
        SimpleNamespace(text="bonjour", t0_ms=100, t1_ms=300),
        SimpleNamespace(text="le", t0_ms=30100, t1_ms=30300),
        SimpleNamespace(text="monde", t0_ms=60300, t1_ms=60700),
    ]
    monkeypatch.setattr(transcribe_cpp, "align_words", lambda **_kwargs: tuple(aligned))
    result = transcribe_cpp.optionally_align_words(response, request=request(85), language="fr")
    assert result.segments == segments
    assert len(result.segments) == 3
    assert result.words == tuple(aligned)


def test_no_segments_falls_back_to_acoustically_aligned_word_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUD9_WORD_ALIGNMENT_STRATEGY", "forced")
    monkeypatch.setenv("CLOUD9_WORD_ALIGNMENT_URL", "internal")
    response = SimpleNamespace(text="bonjour", language="fr", segments=(), words=())
    aligned = (SimpleNamespace(text="bonjour", t0_ms=100, t1_ms=400),)
    monkeypatch.setattr(transcribe_cpp, "align_words", lambda **_kwargs: aligned)
    result = transcribe_cpp.optionally_align_words(response, request=request(3), language="fr")
    assert len(result.segments) == 1
    assert result.segments[0].t0_ms == 100
    assert result.segments[0].t1_ms == 400
