from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import patch

from acoustic_e2e import assert_port_is_free, encode_multipart, validate_alignment
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


def test_windows_loopback_timeout_allows_real_bind_attempt() -> None:
    with patch("acoustic_e2e.socket.create_connection", side_effect=TimeoutError("timed out")):
        assert_port_is_free(39087)


def test_existing_loopback_service_rejected_without_false_certification() -> None:
    with (
        patch("acoustic_e2e.socket.create_connection", return_value=nullcontext()),
        pytest.raises(RuntimeError, match="already in use"),
    ):
        assert_port_is_free(39087)
