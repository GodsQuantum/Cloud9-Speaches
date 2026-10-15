#!/usr/bin/env python3
"""Strict Vulkan-vs-CPU Cloud9 ASR benchmark (inside ephemeral CT410 container).

Host launcher MUST pass host preflight before invoking Docker.
Never download weights or access a different host.
"""

import argparse
import json
import math
from pathlib import Path
import time

import soundfile as sf

CACHE = Path("/home/ubuntu/.cache/huggingface/hub")
MODELS = {
    "nemotron-q8": ("models--handy-computer--nemotron-3.5-asr-streaming-0.6b-gguf", "Q8_0"),
    "qwen3-1.7b-q8": ("models--handy-computer--Qwen3-ASR-1.7B-gguf", "Q8_0"),
    "qwen3-1.7b-q5": ("models--handy-computer--Qwen3-ASR-1.7B-gguf", "Q5_K_M"),
    "qwen3-1.7b-q4": ("models--handy-computer--Qwen3-ASR-1.7B-gguf", "Q4_K_M"),
    "nemotron-q5": ("models--handy-computer--nemotron-3.5-asr-streaming-0.6b-gguf", "Q5_K_M"),
    "nemotron-q4": ("models--handy-computer--nemotron-3.5-asr-streaming-0.6b-gguf", "Q4_K_M"),
}


def words_valid(words: list, transcript: str, duration: float) -> bool:
    if len(words) != len(transcript.split()) or not words:
        return False
    prev = 0.0
    for item in words:
        try:
            text, start, end = item.text, item.t0_ms / 1000, item.t1_ms / 1000
        except (AttributeError, ValueError, TypeError):
            return False
        if not text.strip() or not math.isfinite(start) or not math.isfinite(end):
            return False
        if not (0 <= start < end <= duration + 0.02) or start < prev - 0.04:
            return False
        prev = end
    return True


def load_audio_16k(path: Path | str) -> tuple:
    """Return mono float32 16 kHz, source rate, inference rate and duration.

    Do not change the immutable gold WAV: conversion occurs in memory using
    FFmpeg's resampler, already present in Cloud9-Speaches images.
    """
    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim != 1 or rate <= 0:
        raise ValueError(f"Invalid audio shape/sample rate: {audio.shape}, {rate}")
    original_rate = rate
    original_duration = len(audio) / rate
    if rate != 16000:
        import subprocess

        import numpy as np

        converted = subprocess.run(
            [
                "/usr/bin/ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "f32le",
                "pipe:1",
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        if len(converted.stdout) % 4 != 0 or not converted.stdout:
            raise ValueError("ffmpeg returned truncated or empty 16 kHz PCM")
        audio = np.frombuffer(converted.stdout, dtype="<f4").copy()
        rate = 16000
    duration = len(audio) / rate
    if abs(duration - original_duration) > 0.005:
        raise ValueError("Resampling changed the audio duration unexpectedly")
    return audio, original_rate, rate, duration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "vulkan"], required=True)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--audio-check-only", action="store_true", help="Validate 16 kHz audio input without loading a model"
    )
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5:
        parser.error("repeats must be 1..5")
    try:
        audio, original_rate, rate, duration = load_audio_16k("/bench/fr_regression_v1.wav")
    except ValueError as error:
        parser.error(str(error))
    if args.audio_check_only:
        print(
            json.dumps(
                {
                    "source_rate_hz": original_rate,
                    "inference_rate_hz": rate,
                    "duration_s": duration,
                    "samples": len(audio),
                }
            )
        )
        return
    repo, quant = MODELS[args.model]
    paths = list((CACHE / repo / "snapshots").glob(f"*/*-{quant}.gguf"))
    if len(paths) != 1:
        parser.error(f"exact cached GGUF missing: {args.model}; do not download implicitly")
    from transcribe_cpp import Model  # pyrefly: ignore[missing-import] — installed in runtime Docker image

    started = time.monotonic()
    model = Model(str(paths[0]), backend=args.backend)
    loaded = time.monotonic()
    passes = []
    with model.session() as session:
        for i in range(args.repeats):
            began = time.monotonic()
            result = session.run(
                audio,
                language=None if args.model.startswith("qwen3") else "fr-FR",
                timestamps="none" if args.model.startswith("qwen3") else "word",
            )
            seconds = time.monotonic() - began
            word_rows = list(result.words)
            passes.append(
                {
                    "pass": i + 1,
                    "seconds": round(seconds, 5),
                    "rtf": round(seconds / duration, 5),
                    "words_native_valid": words_valid(word_rows, result.text, duration) if word_rows else False,
                    "native_word_rows": len(word_rows),
                    "text": result.text,
                }
            )
    print(
        "CLOUD9_VULKAN_ASR_AB="
        + json.dumps(
            {
                "model": args.model,
                "gguf_bytes": paths[0].stat().st_size,
                "backend": args.backend,
                "source_audio_rate_hz": original_rate,
                "inference_rate_hz": rate,
                "audio_seconds": duration,
                "load_seconds": round(loaded - started, 5),
                "passes": passes,
                "hardware": "Cloud9 Ryzen 7 8845HS 780M",
                "note": "Do not compare WER with other corpora, and do not treat monotonic timings as manual-gold accuracy.",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
