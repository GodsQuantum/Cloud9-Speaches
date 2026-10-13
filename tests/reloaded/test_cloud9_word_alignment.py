"""No-GPU contract tests for existing AutoSubs forced-alignment integration."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from speaches.executors.cloud9_word_alignment import (
    WordAlignmentError,
    align_words,
    native_word_timings_valid,
    optionally_align_whisper_words,
)
from speaches.executors.transcribe_cpp import optionally_align_words


def fake_audio() -> SimpleNamespace:
    return SimpleNamespace(
        data=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        duration=1.0,
    )


def response(words: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"word_segments": words},
    )


WORDS = [
    {"word": "Bonjour,", "start": 0.02, "end": 0.4, "score": 0.98},
    {"word": "monde.", "start": 0.46, "end": 0.88, "score": 0.87},
]


def test_real_timing_format_includes_ms_and_confidence() -> None:
    with patch("speaches.executors.cloud9_word_alignment.httpx.post", return_value=response(WORDS)) as mock:
        got = align_words(
            audio=fake_audio(), transcript="Bonjour, monde.", language="fr", endpoint="http://aligner/align"
        )
    assert [(w.text, w.t0_ms, w.t1_ms, w.score) for w in got] == [
        ("Bonjour,", 20, 400, 0.98),
        ("monde.", 460, 880, 0.87),
    ]
    assert mock.call_args.kwargs["data"]["transcript"] == "Bonjour, monde."
    assert mock.call_args.kwargs["files"]["file"][2] == "audio/wav"


@pytest.mark.parametrize(
    "bad",
    [
        [{**WORDS[0], "end": 0.0}, WORDS[1]],
        [WORDS[0], {**WORDS[1], "start": 0.1}],
        [WORDS[0]],
        [WORDS[0], {**WORDS[1], "end": 30.0}],
        [WORDS[0], {**WORDS[1], "score": -1.0}],
        [WORDS[0], {**WORDS[1], "word": "autre."}],
    ],
)
def test_invalid_and_incomplete_timings_fail_closed(bad: list[dict]) -> None:
    with (
        patch("speaches.executors.cloud9_word_alignment.httpx.post", return_value=response(bad)),
        pytest.raises(WordAlignmentError),
    ):
        align_words(audio=fake_audio(), transcript="Bonjour, monde.", language="fr", endpoint="http://aligner/align")


def test_no_aligner_requested_leaves_existing_engine_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUD9_WORD_ALIGNMENT_URL", raising=False)
    monkeypatch.delenv("CLOUD9_WORD_ALIGNMENT_STRICT", raising=False)
    result = SimpleNamespace(text="Bonjour, monde.", language="fr", segments=(), words=())
    request = SimpleNamespace(audio=fake_audio(), response_format="verbose_json", timestamp_granularities=["word"])
    assert optionally_align_words(result, request=request, language="fr") is result


def test_qwen_no_native_timestamp_gets_aligned_without_rewriting_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUD9_WORD_ALIGNMENT_URL", "http://aligner/align")
    monkeypatch.setenv("CLOUD9_WORD_ALIGNMENT_STRATEGY", "native-first")
    result = SimpleNamespace(text="Bonjour, monde.", language="fr", segments=(), words=())
    request = SimpleNamespace(
        audio=fake_audio(),
        language="fr",
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )
    with patch(
        "speaches.executors.transcribe_cpp.align_words",
        return_value=(
            SimpleNamespace(text="Bonjour,", t0_ms=20, t1_ms=400),
            SimpleNamespace(text="monde.", t0_ms=460, t1_ms=880),
        ),
    ) as mocked:
        aligned = optionally_align_words(result, request=request, language=None)
    assert aligned.text == result.text
    assert len(aligned.words) == 2
    assert aligned.segments[0].t0_ms == 20
    assert aligned.segments[0].t1_ms == 880
    assert mocked.call_args.kwargs["language"] == "fr"


def test_native_word_timing_is_preserved_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUD9_WORD_ALIGNMENT_URL", "http://aligner/align")
    monkeypatch.setenv("CLOUD9_WORD_ALIGNMENT_STRATEGY", "native-first")
    native = (SimpleNamespace(text="Bonjour", t0_ms=4, t1_ms=350),)
    result = SimpleNamespace(text="Bonjour", language="fr", segments=(), words=native)
    request = SimpleNamespace(audio=fake_audio(), response_format="verbose_json", timestamp_granularities=["word"])
    with patch("speaches.executors.transcribe_cpp.align_words") as mocked:
        assert optionally_align_words(result, request=request, language="fr") is result
    mocked.assert_not_called()


def test_non_word_request_does_not_call_aligner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUD9_WORD_ALIGNMENT_URL", "http://aligner/align")
    result = SimpleNamespace(text="Bonjour", language="fr", segments=(), words=())
    request = SimpleNamespace(audio=fake_audio(), response_format="json", timestamp_granularities=["word"])
    with patch("speaches.executors.transcribe_cpp.align_words") as mocked:
        assert optionally_align_words(result, request=request, language="fr") is result
    mocked.assert_not_called()


def test_strict_no_service_refuses_fabricated_words(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUD9_WORD_ALIGNMENT_URL", raising=False)
    monkeypatch.setenv("CLOUD9_WORD_ALIGNMENT_STRICT", "1")
    result = SimpleNamespace(text="Bonjour", language="fr", segments=(), words=())
    request = SimpleNamespace(audio=fake_audio(), response_format="verbose_json", timestamp_granularities=["word"])
    with pytest.raises(WordAlignmentError, match="No word timestamps"):
        optionally_align_words(result, request=request, language="fr")


def test_actual_nemotron_style_trailing_empty_and_audio_overrun_rejected() -> None:
    """Observed on Cloud9: 37 native items for 36 spoken words, final item empty/14.24s."""
    native = (
        SimpleNamespace(text="Bonjour,", t0_ms=320, t1_ms=1200),
        SimpleNamespace(text="monde.", t0_ms=1200, t1_ms=1520),
        SimpleNamespace(text="", t0_ms=14160, t1_ms=14240),
    )
    assert not native_word_timings_valid(native, transcript="Bonjour, monde.", duration=14.17)
    assert not native_word_timings_valid(native[:2], transcript="Bonjour, monde.", duration=1.0)


def test_invalid_native_words_force_existing_autosubs_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUD9_WORD_ALIGNMENT_URL", "http://aligner/align")
    monkeypatch.setenv("CLOUD9_WORD_ALIGNMENT_STRATEGY", "native-first")
    native = (
        SimpleNamespace(text="Bonjour,", t0_ms=20, t1_ms=400),
        SimpleNamespace(text="monde.", t0_ms=460, t1_ms=880),
        SimpleNamespace(text="", t0_ms=900, t1_ms=1050),
    )
    result = SimpleNamespace(
        text="Bonjour, monde.",
        language="fr",
        segments=(SimpleNamespace(text="Bonjour, monde.", t0_ms=0, t1_ms=1050),),
        words=native,
    )
    request = SimpleNamespace(
        audio=fake_audio(),
        language="fr",
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )
    aligned = (
        SimpleNamespace(text="Bonjour,", t0_ms=20, t1_ms=400),
        SimpleNamespace(text="monde.", t0_ms=460, t1_ms=880),
    )
    with patch("speaches.executors.transcribe_cpp.align_words", return_value=aligned) as mocked:
        out = optionally_align_words(result, request=request, language="fr")
    mocked.assert_called_once()
    assert len(out.words) == 2
    assert out.segments[0].t1_ms == 880


def test_whisper_turbo_word_timestamps_native_pass_without_extra_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai.types.audio

    monkeypatch.setenv("CLOUD9_WORD_ALIGNMENT_URL", "http://aligner/align")
    monkeypatch.setenv("CLOUD9_WORD_ALIGNMENT_STRATEGY", "native-first")
    original = openai.types.audio.TranscriptionVerbose(
        language="fr",
        duration=1.0,
        text="Bonjour monde.",
        segments=[],
        words=[
            openai.types.audio.TranscriptionWord(start=0.02, end=0.43, word="Bonjour"),
            openai.types.audio.TranscriptionWord(start=0.46, end=0.88, word="monde."),
        ],
    )
    request = SimpleNamespace(
        audio=fake_audio(),
        response_format="verbose_json",
        timestamp_granularities=["word"],
        language="fr",
    )
    with patch("speaches.executors.cloud9_word_alignment.align_words") as mocked:
        out = optionally_align_whisper_words(original, request=request)
    assert out is original
    mocked.assert_not_called()


def test_whisper_turbo_invalid_words_get_real_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai.types.audio

    monkeypatch.setenv("CLOUD9_WORD_ALIGNMENT_URL", "http://aligner/align")
    original = openai.types.audio.TranscriptionVerbose(
        language="fr",
        duration=1.0,
        text="Bonjour monde.",
        segments=[],
        words=[
            openai.types.audio.TranscriptionWord(start=0.02, end=0.43, word="Bonjour"),
            openai.types.audio.TranscriptionWord(start=0.90, end=1.24, word="monde."),
        ],
    )
    aligned = (
        SimpleNamespace(text="Bonjour", t0_ms=20, t1_ms=400),
        SimpleNamespace(text="monde.", t0_ms=460, t1_ms=880),
    )
    request = SimpleNamespace(
        audio=fake_audio(),
        response_format="verbose_json",
        timestamp_granularities=["word"],
        language="fr",
    )
    with patch("speaches.executors.cloud9_word_alignment.align_words", return_value=aligned):
        out = optionally_align_whisper_words(original, request=request)
    assert out.text == original.text
    assert out.words is not None
    assert [w.end for w in out.words] == [0.4, 0.88]
    assert original.words is not None
    assert original.words[1].end == 1.24
