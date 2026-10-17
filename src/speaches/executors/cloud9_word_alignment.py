"""Strict acoustic alignment via native Vulkan CrispASR or HTTP compatibility.

Never synthesize timestamps or confidence scores; native execution is selected
with CLOUD9_WORD_ALIGNMENT_URL=internal.
"""

from __future__ import annotations

from io import BytesIO
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any

import httpx
import soundfile as sf

MAX_ALIGN_SECONDS = 21600.0  # six hours, with acoustic chunk offsets required


class WordAlignmentError(RuntimeError):
    """No trustworthy alignment is available. Never manufacture word times."""


def lexical_key(value: str) -> str:
    """Compare orthographic words without altering the published transcript."""
    return value.strip().strip(' .,!?;:«»()[]{}"“”').casefold()


def spoken_tokens(transcript: str) -> list[str]:
    """Track CJK ideographs individually and attach punctuation to spoken words."""
    result: list[str] = []
    prefix = ""
    for piece in transcript.split():
        buffer = ""
        for ch in piece:
            if 0x3400 <= ord(ch) <= 0x9FFF:
                if buffer:
                    result.append(prefix + buffer)
                    prefix = ""
                    buffer = ""
                result.append(prefix + ch)
                prefix = ""
            elif ch in "，。！？、；：" and result:
                if buffer:
                    result.append(prefix + buffer)
                    prefix = ""
                    buffer = ""
                result[-1] += ch
            else:
                buffer += ch
        if buffer and any(ch.isalnum() for ch in buffer):
            result.append(prefix + buffer)
            prefix = ""
        elif buffer and result:
            result[-1] += buffer
        else:
            prefix += buffer
    if prefix or not result:
        raise WordAlignmentError("Transcript has no attachable spoken tokens")
    return result


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
    segments: Any = None,
    model: str | None = None,
) -> tuple[SimpleNamespace, ...]:
    """Return actual acoustic per-word timings, or fail explicitly."""
    if not transcript.strip():
        raise WordAlignmentError("Cannot align an empty transcript")
    if not language.strip():
        raise WordAlignmentError("Explicit language required for forced alignment")
    if endpoint != "internal" and not endpoint.startswith(("http://", "https://")):
        raise WordAlignmentError("Alignment URL must be internal or HTTP(S)")
    duration = float(audio.duration)
    if not math.isfinite(duration) or not 0 < duration <= MAX_ALIGN_SECONDS:
        raise WordAlignmentError("Audio must be 0-21600 seconds")

    form = {"transcript": transcript, "language": language}
    if model and model not in ("auto", "default"):
        form["model"] = model
    if duration > 38.0:
        if not segments:
            raise WordAlignmentError(
                "Long audio needs ASR segment offsets; refusing guessed word timing"
            )
        groupings = []
        for idx, seg in enumerate(segments):
            try:
                t0 = getattr(seg, "t0_ms", None)
                t1 = getattr(seg, "t1_ms", None)
                left = float(t0) / 1000 if t0 is not None else float(seg.start)
                right = float(t1) / 1000 if t1 is not None else float(seg.end)
                groupings.append({"start": left, "end": right, "text": seg.text})
            except (AttributeError, TypeError, ValueError) as exc:
                raise WordAlignmentError(f"Invalid long-form ASR segment #{idx}") from exc
        from speaches.executors.cloud9_alignment_groups import groups_for_alignment
        try:
            groups_for_alignment(transcript, duration, json.dumps(groupings, ensure_ascii=False))
        except ValueError as exc:
            raise WordAlignmentError(f"Unsafe long-form ASR segmentation: {exc}") from exc
        form["segments_json"] = json.dumps(groupings, ensure_ascii=False)

    wav = BytesIO()
    sf.write(wav, audio.data, audio.sample_rate, format="WAV", subtype="PCM_16")
    try:
        if endpoint == "internal":
            from speaches.executors.cloud9_ctc import run_alignment
            with tempfile.TemporaryDirectory(prefix="speaches-native-ctc-") as tmp:
                source = Path(tmp) / "speech.wav"
                source.write_bytes(wav.getvalue())
                payload = run_alignment(
                    source, transcript, language, form.get("model"),
                    form.get("segments_json"),
                )
        else:
            response = httpx.post(
                endpoint,
                files={"file": ("speech.wav", wav.getvalue(), "audio/wav")},
                data=form,
                timeout=httpx.Timeout(7200.0, connect=5.0),
            )
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError, RuntimeError, OSError) as exc:
        raise WordAlignmentError("Forced acoustic alignment failed or unavailable") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("word_segments"), list):
        raise WordAlignmentError("Forced-alignment response lacks word_segments")
    rows = payload["word_segments"]
    expected = spoken_tokens(transcript)
    if len(rows) != len(expected):
        raise WordAlignmentError(
            f"Forced alignment incomplete: {len(rows)} words for {len(expected)} spoken tokens"
        )
    words: list[SimpleNamespace] = []
    previous_end = 0.0
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("word"), str) or not row["word"].strip():
            raise WordAlignmentError(f"Invalid aligned word {i}")
        # Keep the recognized text, not a substitute supplied by an aligner.
        if lexical_key(row["word"]) != lexical_key(expected[i]):
            raise WordAlignmentError(f"Lexical mismatch in aligned word {i}")
        try:
            start = float(row["start"])
            end = float(row["end"])
            # CrispASR Vulkan does not expose frame confidence. Preserve this
            # uncertainty; never fabricate a score to satisfy a legacy client.
            score = float(row["score"]) if row.get("score") is not None else None
        except (KeyError, TypeError, ValueError) as exc:
            raise WordAlignmentError(f"Invalid timing or confidence for word {i}") from exc
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or (score is not None and (not math.isfinite(score) or not 0 <= score <= 1))
            or start < 0
            or end <= start
            or end > duration + 0.15
            or start < previous_end - 0.04
        ):
            raise WordAlignmentError(f"Invalid or nonmonotonic word time {i}")
        words.append(SimpleNamespace(text=row["word"], t0_ms=round(start * 1000), t1_ms=round(end * 1000), score=score))
        previous_end = end
    if not words:
        raise WordAlignmentError("No word timings returned")
    return tuple(words)


def optionally_align_whisper_words(response: Any, *, request: Any) -> Any:
    """Preserve Whisper CT2/native words or use the same CPU aligner when enabled.

    This covers large-v3, large-v3-turbo and small without changing the
    upstream faster-whisper decoder. The default (unset URL) is a no-op.
    """
    if request.response_format != "verbose_json" or "word" not in request.timestamp_granularities:
        return response
    strategy = os.getenv("CLOUD9_WORD_ALIGNMENT_STRATEGY", "native-first")
    if strategy not in ("native-first", "forced"):
        raise WordAlignmentError(f"Invalid alignment strategy: {strategy}")
    native = tuple(
        SimpleNamespace(
            text=word.word,
            t0_ms=round(word.start * 1000),
            t1_ms=round(word.end * 1000),
        )
        for word in (response.words or [])
    )
    valid_native = native_word_timings_valid(native, transcript=response.text, duration=request.audio.duration)
    if valid_native and strategy == "native-first":
        return response

    endpoint = os.getenv("CLOUD9_WORD_ALIGNMENT_URL", "").strip()
    if not endpoint:
        if os.getenv("CLOUD9_WORD_ALIGNMENT_STRICT") == "1":
            raise WordAlignmentError("Whisper word timestamps invalid and no alignment service configured")
        return response

    words = align_words(
        audio=request.audio,
        transcript=response.text,
        language=(request.language or response.language or "").split("-", 1)[0].lower(),
        endpoint=endpoint,
        segments=getattr(response, "segments", None),
        model=getattr(request, "alignment_model", None),
    )
    import openai.types.audio

    return response.model_copy(
        update={
            "words": [
                openai.types.audio.TranscriptionWord(
                    start=word.t0_ms / 1000,
                    end=word.t1_ms / 1000,
                    word=word.text,
                )
                for word in words
            ],
        }
    )
