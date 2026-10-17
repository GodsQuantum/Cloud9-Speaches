"""Cloud9 native CTC regression: reject ungrounded timings and preserve Han tokens."""

import json

import pytest

from speaches.executors.cloud9_alignment_groups import groups_for_alignment
from speaches.executors.cloud9_ctc import acoustic_tokens, restore_original_tokens, selected_compute_backend
from speaches.executors.cloud9_word_alignment import spoken_tokens


def test_chinese_acoustic_tokens_not_single_utterance() -> None:
    assert acoustic_tokens("你好世界。 OpenAI") == ["你", "好", "世", "界。", "OpenAI"]
    assert spoken_tokens("你好世界。 OpenAI") == ["你", "好", "世", "界。", "OpenAI"]


def test_longform_requires_real_asr_chunk_offsets() -> None:
    with pytest.raises(ValueError, match="long audio requires"):
        groups_for_alignment("bonjour à toutes et à tous", 104.96, None)


def test_mismatched_transcript_rejected() -> None:
    rows = [{"start": 0, "end": 8, "text": "Bonjour les amis"}]
    with pytest.raises(ValueError, match="differ"):
        groups_for_alignment("Bonsoir les amis", 8.0, json.dumps(rows))


def test_hyphen_preserves_measured_start_end() -> None:
    raw = [{"word": "Tais-", "start": 0.10, "end": 0.36}, {"word": "toi", "start": 0.38, "end": 0.72}]
    assert restore_original_tokens(raw, "Tais-toi") == [{"word": "Tais-toi", "start": 0.10, "end": 0.72}]


def test_missing_acoustic_word_must_fail() -> None:
    with pytest.raises(ValueError, match=r"omitted|mismatch"):
        restore_original_tokens([{"word": "Bonjour", "start": 0.1, "end": 0.5}], "Bonjour les amis")


@pytest.mark.parametrize("backend", ["auto", "cpu", "cuda", "vulkan", "metal"])
def test_ctc_backend_selection_all_platforms(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    monkeypatch.setenv("CLOUD9_CTC_BACKEND", backend.upper())
    assert selected_compute_backend() == backend


def test_ctc_backend_rejects_unsupported_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUD9_CTC_BACKEND", "vulkan-and-guess")
    with pytest.raises(ValueError, match="Invalid CLOUD9_CTC_BACKEND"):
        selected_compute_backend()
