from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pyannote.core import Annotation


def extract_diarization_annotation(result: object) -> Annotation:
    annotation = getattr(result, "speaker_diarization", result)
    if not callable(getattr(annotation, "itertracks", None)):
        raise TypeError(f"Unexpected diarization result without tracks: {type(result).__name__}")
    return cast("Annotation", annotation)
