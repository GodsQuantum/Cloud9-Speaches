"""CPU-only smoke for existing AutoSubs service (run in isolated test container)."""

from itertools import pairwise
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import soundfile as sf

from speaches.executors.cloud9_word_alignment import align_words
from speaches.executors.transcribe_cpp import optionally_align_words

audio_path = Path("/bench/fr_regression_v1.wav")
reference = Path("/bench/fr_regression_v1.txt").read_text().strip()
samples, sample_rate = sf.read(str(audio_path), dtype="float32")
audio = SimpleNamespace(data=samples, sample_rate=sample_rate, duration=len(samples) / sample_rate)

words = align_words(
    audio=audio,
    transcript=reference,
    language="fr",
    endpoint="http://192.168.1.219:8006/align",
)
assert len(words) == len(reference.split()) == 37
assert all(w.t1_ms > w.t0_ms for w in words)
assert all(a.t1_ms <= b.t0_ms + 40 for a, b in pairwise(words))
print("REAL_CPU_WORD_ALIGN_PASS", len(words), words[0].t0_ms, words[-1].t1_ms, round(mean(w.score for w in words), 3))

result = SimpleNamespace(text=reference, language="fr", segments=(), words=())
request = SimpleNamespace(
    audio=audio,
    language="fr",
    response_format="verbose_json",
    timestamp_granularities=["word"],
)
out = optionally_align_words(result, request=request, language=None)
assert out.text == reference
assert len(out.words) == 37
assert out.segments[0].t0_ms == out.words[0].t0_ms
assert out.segments[0].t1_ms == out.words[-1].t1_ms
print("QWEN_NO_NATIVE_WORD_TIMING_FALLBACK_PASS", len(out.words))
