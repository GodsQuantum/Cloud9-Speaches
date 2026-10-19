#!/usr/bin/env python3
"""Compare existing cached Cloud9 ASR models using the SAME CPU-only audio fixture.

Use inside a constrained Docker test container, never Vulkan when GPU is wedged.
Output JSON single line to facilitate append-only historical benchmark ledger.
"""

import argparse
import json
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
import soundfile as sf

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT = Path("/home/ubuntu/.cache/huggingface/hub")


class TimedWord(TypedDict):
    text: str
    start: float
    end: float


def normalized(text: str) -> list[str]:
    import re
    import unicodedata

    text = "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text).strip().split()


def wer(reference: str, hypothesis: str) -> tuple[float, int, int]:
    a, b = normalized(reference), normalized(hypothesis)
    d = list(range(len(b) + 1))
    for i, w in enumerate(a, 1):
        new = [i]
        for j, h in enumerate(b, 1):
            new.append(min(new[j - 1] + 1, d[j] + 1, d[j - 1] + (w != h)))
        d = new
    return d[-1] / max(1, len(a)), d[-1], len(a)


def load_audio() -> tuple[np.ndarray, float, str]:
    p = Path("/bench/fr_regression_v1.wav")
    audio, sr = sf.read(str(p), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly

        gcd = math.gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // gcd, sr // gcd).astype(np.float32)
    return audio, len(audio) / 16000, Path("/bench/fr_regression_v1.txt").read_text().strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("turbo-ct2", "qwen3-1.7b-q8", "nemotron-q8"), required=True)
    parser.add_argument("--beam-size", type=int, choices=(1, 3, 5), default=5)
    args = parser.parse_args()
    audio, duration, reference = load_audio()
    start = time.monotonic()
    words: list[TimedWord] = []
    if args.engine == "turbo-ct2":
        from faster_whisper import WhisperModel

        snapshots = sorted((ROOT / "models--deepdml--faster-whisper-large-v3-turbo-ct2/snapshots").iterdir())
        assert len(snapshots) == 1
        engine = WhisperModel(str(snapshots[0]), device="cpu", compute_type="int8", cpu_threads=2, num_workers=1)
        loaded = time.monotonic()
        segments, _info = engine.transcribe(
            audio,
            language="fr",
            beam_size=args.beam_size,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
            temperature=0.0,
        )
        segments = list(segments)
        hypothesis = "".join(seg.text for seg in segments).strip()
        words = [
            {"text": w.word.strip(), "start": w.start, "end": w.end} for seg in segments for w in (seg.words or [])
        ]
    else:
        from transcribe_cpp import Model  # pyrefly: ignore[missing-import] — installed in runtime Docker image

        label = (
            "models--handy-computer--Qwen3-ASR-1.7B-gguf"
            if args.engine == "qwen3-1.7b-q8"
            else "models--handy-computer--nemotron-3.5-asr-streaming-0.6b-gguf"
        )
        paths = list((ROOT / label / "snapshots").glob("*/*-Q8_0.gguf"))
        assert len(paths) == 1, paths
        engine = Model(str(paths[0]), backend="cpu")
        loaded = time.monotonic()
        with engine.session() as session:
            output = session.run(
                cast("Sequence[float]", audio),
                language=None if args.engine.startswith("qwen3") else "fr-FR",
                timestamps="none" if args.engine.startswith("qwen3") else "word",
            )
        hypothesis = output.text
        words = [{"text": w.text, "start": w.t0_ms / 1000, "end": w.t1_ms / 1000} for w in output.words]
    finished = time.monotonic()
    valid_words = all(
        bool(w["text"])
        and math.isfinite(w["start"])
        and math.isfinite(w["end"])
        and 0 <= w["start"] < w["end"] <= duration + 0.02
        for w in words
    ) and len(words) == len(hypothesis.split())
    error, edits, nref = wer(reference, hypothesis)
    result = {
        "engine": args.engine,
        "beam_size": args.beam_size if args.engine == "turbo-ct2" else None,
        "audio_seconds": duration,
        "device": "cpu-strict",
        "load_seconds": round(loaded - start, 5),
        "inference_seconds": round(finished - loaded, 5),
        "rtf": round((finished - loaded) / duration, 5),
        "wer": error,
        "edits": edits,
        "ref_words": nref,
        "hypothesis": hypothesis,
        "native_word_count": len(words),
        "native_words_valid": valid_words,
        "native_word_first_last": [words[0], words[-1]] if words else [],
        "source": "one synthetic French fixture; NOT a podcast or gold word-timing certification",
    }
    print("CLOUD9_CACHED_ASR_CPU_AB=" + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
