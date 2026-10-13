# Cloud9-Speaches ↔ AutoSubs word-for-word timing (13 Oct 2026)

**Status: experimental fork integration; production unchanged.** Validated against the user's real Cloud9 AutoSubs WhisperX aligner, not a generic Speaches deployment.

## Existing infrastructure

- STT: Cloud9-Speaches (CT410 `192.168.1.219:8005`), transcribe.cpp Qwen3-ASR/Nemotron GGUF Vulkan; Whisper CPU fallback, plus TTS and diarization.
- **Already installed** CPU WhisperX forced-alignment service: `autosubs-aligner`, `autosubs-whisperx-aligner:3.8.6-v1`, `192.168.1.219:8006`, `POST /align`; models `fr:default`, `ALIGN_DEVICE=cpu`, four threads, healthy.
- AutoSubs GitHub source of truth: https://github.com/GodsQuantum/AutoSubs . **Do not create another aligner, rename these services or change paths.**
- Source limitation: `transcribe_cpp.resolve_timestamp_mode` correctly emits `none` for Qwen3-ASR: this GGUF model does **not** natively support word timestamps. Never falsify native support.

## Real CPU aligner smoke

Reference fixture `bench/fixtures/fr_regression_v1.wav`, 14.169977 s French Piper synthesis.

| Input transcript | Word intervals returned | Timings | Mean confidence |
|---|---:|---|---:|
| Reference text (37 orthographic tokens) | 37 | monotonically valid, 0.260–13.950 s | 0.874 |
| Actual Nemotron text (36 tokens, imperfect proper names) | 36 | monotonically valid, 0.260–13.950 s | 0.879 |

Reference alignment completed in **4.994 s** HTTP. No word intervals were empty, negative, reversed, or beyond audio length. HTTP 200 on endpoints :8090, :8091, :8005 and :8006 after the test.

**Important:** these are validity/coverage/timing-consistency checks, **not** a measured millisecond accuracy claim. Real ground-truth word boundary annotations and podcast audio test clips are still needed. Forced alignment cannot fix wrong *words* emitted by the ASR model. For publication-quality French subtitles, judge both textual WER and word boundary timing error, including technical proper nouns, silent gaps, word hyphens, speech overlaps, punctuation and re-listening.

## Proposed opt-in Cloud9-Speaches bridge

In this branch only: `src/speaches/executors/cloud9_word_alignment.py` validates existing AutoSubs response and rejects incomplete/unordered/unreasonable data rather than evenly distributing words. `transcribe_cpp.py` calls the bridge **only** if:
- non-streaming STT;
- explicit `response_format=verbose_json`;
- requested `timestamp_granularities[]=word`;
- the service URL is explicitly configured, and either no native words are available or operator requests forced alignment.

Suggested opt-in environment *after all gates pass*:

```ini
CLOUD9_WORD_ALIGNMENT_URL=http://192.168.1.219:8006/align
CLOUD9_WORD_ALIGNMENT_STRATEGY=native-first
CLOUD9_WORD_ALIGNMENT_STRICT=1
```

Use `CLOUD9_WORD_ALIGNMENT_STRATEGY=forced` when AutoSubs prefers alignment even on native word timestamp-capable Nemotron/Parakeet; compare manually marked boundaries before selecting a global strategy. No live configuration has been changed.

The response uses existing OpenAI `verbose_json.words[]` (`word`, `start`, `end`), retaining original ASR words. Qwen3-ASR still runs with its original `timestamps=none` and the fallback aligns its text after transcription. For results without trustworthy native segment times, a new segment is derived from the first/last *real* aligned word. The service returns HTTP 503 if the selected aligner is unavailable or responds with invalid timing; it does not make up timestamps. Standard `json`, streaming text, TTS, native words and other routes remain unchanged.

Audio must be <=300 seconds for one alignment request; AutoSubs must chunk long recordings, track absolute segment offsets and merge results. **Never silently truncate a podcast**, and do not try to align unbounded multi-hour media in one API request.

## Further R&D (not yet installed or benchmarked)

- Official [Qwen3-ForcedAligner-0.6B](https://github.com/QwenLM/Qwen3-ASR) supports French word timings on arbitrary ASR text, with a 5 minute ceiling. It is an independent *candidate* against the already working CPU WhisperX aligner. Do not replace WhisperX until rigorous matched-audio CPU/Vulkan hardware A/B and accuracy gates.
- Native transcribe.cpp Nemotron/Parakeet word timings can be tested with `--timestamps word`; this does not mean Qwen3-ASR supports timestamps. v0.4.0 native + bindings are a separate staged upgrade that still requires Vulkan validation.
- Accurate podcast captions require both text accuracy and measured word boundary accuracy; use manually labeled French podcasts including accent, speech overlaps, silence, césure/hyphen and multiple speakers.

## Release gate

1. This branch: Ruff, format, typing and pytest offline; CPU aligner A/B/test cases with actual WordAlignment API.
2. Confirm the AutoSubs client consumes `verbose_json.words` or its own `/align` output with correct absolute offsets and confidence; never replace its existing pipeline based on assumption.
3. Test on real French subtitle media, with manually annotated word times and error thresholds; feature flag to revert.
4. Restore healthy GPU before any new Vulkan engine/0.4.0 benchmark; reject while the CT700 Pym benchmark PID and GTT remain hung.
5. Merge only after all prod tests, keep GHCR image name compatibility, preserve stable runtime and rollback path.
