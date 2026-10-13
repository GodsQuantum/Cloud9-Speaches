"""Optional, verified CPU forced-alignment bridge for Cloud9-Speaches.

This calls the *existing* AutoSubs alignment service. It never infers word
timings by spacing words equally across a segment.
"""

from __future__ import annotations

from io import BytesIO
import math
from types import SimpleNamespace
from typing import Any

import httpx
import soundfile as sf

MAX_ALIGN_SECONDS = 300.0  # larger podcasts must be chunked by AutoSubs


class WordAlignmentError(RuntimeError):
    """No trustworthy alignment is available. Never manufacture word times."""


def lexical_key(value: str) -> str:
    """Compare orthographic words without altering the published transcript."""
    return value.strip().strip(' .,!?;:«»()[]{}"“”').casefold()


def native_word_timings_valid(words: tuple[Any, ...], *, transcript: str, duration: float) -> bool:
    """Reject empty/native-overrun/incorrect words before accepting native timing."""
    expected = transcript.split()
    if not words or len(words) != len(expected):
        return False
    previous_end = 0.0
    for index, word in enumerate(words):
        try:
            start, end = float(word.t0_ms) / 1000, float(word.t1_ms) / 1000
            spoken = str(word.text)
        except (AttributeError, TypeError, ValueError):
            return False
        if (
            not spoken.strip()
            or lexical_key(spoken) != lexical_key(expected[index])
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or end > duration + 0.02
            or start < previous_end - 0.04
        ):
            return False
        previous_end = end
    return True


def align_words(
    *,
    audio: Any,
    transcript: str,
    language: str,
    endpoint: str,
) -> tuple[SimpleNamespace, ...]:
    """Return actual acoustic per-word timings, or fail explicitly."""
    if not transcript.strip():
        raise WordAlignmentError("Cannot align an empty transcript")
    if not language.strip():
        raise WordAlignmentError("Explicit language required for forced alignment")
    if not endpoint.startswith(("http://", "https://")):
        raise WordAlignmentError("Configured alignment URL must use HTTP(S)")
    duration = float(audio.duration)
    if not math.isfinite(duration) or not 0 < duration <= MAX_ALIGN_SECONDS:
        raise WordAlignmentError("Audio must be 0-300 seconds; chunk longer recordings first")

    wav = BytesIO()
    sf.write(wav, audio.data, audio.sample_rate, format="WAV", subtype="PCM_16")
    try:
        response = httpx.post(
            endpoint,
            files={"file": ("speech.wav", wav.getvalue(), "audio/wav")},
            data={"transcript": transcript, "language": language},
            timeout=httpx.Timeout(120.0, connect=5.0),
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise WordAlignmentError("Forced-alignment service unavailable or invalid") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("word_segments"), list):
        raise WordAlignmentError("Forced-alignment response lacks word_segments")
    rows = payload["word_segments"]
    if len(rows) != len(transcript.split()):
        raise WordAlignmentError(
            f"Forced alignment incomplete: {len(rows)} words for {len(transcript.split())} transcript tokens"
        )
    words: list[SimpleNamespace] = []
    previous_end = 0.0
    expected = transcript.split()
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("word"), str) or not row["word"].strip():
            raise WordAlignmentError(f"Invalid aligned word {i}")
        # Keep the recognized text, not a substitute supplied by an aligner.
        if lexical_key(row["word"]) != lexical_key(expected[i]):
            raise WordAlignmentError(f"Lexical mismatch in aligned word {i}")
        try:
            start = float(row["start"])
            end = float(row["end"])
            score = float(row["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WordAlignmentError(f"Invalid timing or confidence for word {i}") from exc
        if (
            not all(math.isfinite(v) for v in (start, end, score))
            or start < 0
            or end <= start
            or end > duration + 0.15
            or start < previous_end - 0.04
            or not 0 <= score <= 1
        ):
            raise WordAlignmentError(f"Invalid or nonmonotonic word time {i}")
        words.append(SimpleNamespace(text=row["word"], t0_ms=round(start * 1000), t1_ms=round(end * 1000), score=score))
        previous_end = end
    if not words:
        raise WordAlignmentError("No word timings returned")
    return tuple(words)
