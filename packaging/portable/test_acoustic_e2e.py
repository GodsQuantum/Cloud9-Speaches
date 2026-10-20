from __future__ import annotations

from acoustic_e2e import encode_multipart, validate_alignment
import pytest


def test_multipart_contains_real_audio_and_utf8_text() -> None:
    body, content_type = encode_multipart(
        {"transcript": "Bonjour, vingt-trois.", "language": "fr"},
        "audio.wav",
        b"RIFF-WAVE-FIXTURE",
    )
    assert "multipart/form-data" in content_type
    assert b"RIFF-WAVE-FIXTURE" in body
    assert b"vingt-trois" in body
    assert b'name="file"; filename="audio.wav"' in body


def test_acoustic_alignment_enforces_ordered_measured_words() -> None:
    response = {
        "word_segments": [
            {"word": "Bonjour", "start": 0.1, "end": 0.4},
            {"word": "Paris", "start": 0.6, "end": 0.9},
        ],
        "quality": {
            "backend": "cpu",
            "method": "crispasr-ctc-forced-alignment",
            "interpolation": False,
        },
    }
    assert validate_alignment(response, "Bonjour Paris", 1.2) == 2
    response["word_segments"][1]["start"] = 0.3
    with pytest.raises(ValueError, match="monotonic"):
        validate_alignment(response, "Bonjour Paris", 1.2)


def test_acoustic_alignment_rejects_missing_spoken_words() -> None:
    with pytest.raises(ValueError, match="word"):
        validate_alignment(
            {"word_segments": [], "quality": {"backend": "cpu", "interpolation": False}},
            "Bonjour Paris",
            1.0,
        )
