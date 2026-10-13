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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "vulkan"], required=True)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5:
        parser.error("repeats must be 1..5")
    repo, quant = MODELS[args.model]
    paths = list((CACHE / repo / "snapshots").glob(f"*/*-{quant}.gguf"))
    if len(paths) != 1:
        parser.error(f"exact cached GGUF missing: {args.model}; do not download implicitly")
    audio, rate = sf.read("/bench/fr_regression_v1.wav", dtype="float32")
    assert audio.ndim == 1 and rate == 16000, (audio.shape, rate)
    duration = len(audio) / rate
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
