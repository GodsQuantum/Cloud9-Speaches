from __future__ import annotations

from types import SimpleNamespace

import pytest

from speaches.diarization_compat import extract_diarization_annotation


class Annotation:
    def itertracks(self, *, yield_label: bool = False) -> list[object]:
        return [(0, "speaker")] if yield_label else []


def test_pyannote3_annotation_preserved() -> None:
    annotation = Annotation()
    assert extract_diarization_annotation(annotation) is annotation


def test_pyannote4_output_normalized() -> None:
    annotation = Annotation()
    assert extract_diarization_annotation(SimpleNamespace(speaker_diarization=annotation)) is annotation


def test_unknown_output_rejected() -> None:
    with pytest.raises(TypeError, match="diarization"):
        extract_diarization_annotation(object())
