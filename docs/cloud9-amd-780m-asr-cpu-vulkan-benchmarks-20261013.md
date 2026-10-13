# Cloud9-Speaches — Ryzen 7 8845HS / Radeon 780M: CPU vs Vulkan ASR

Last checked: **2026-10-13**. **GPU benchmarks deliberately NOT executed** while host AMDGPU has stuck tasks `D` / elevated GTT. CPU hardware observations and upstream AMD Vulkan reference are identified separately below.

## Cloud9 physical CPU measurements: identical French fixture

All using exactly **14.17s synthetic French audio**, existing cached model weights, Docker 2 vCPU / 5 GiB / no `/dev/dri`, strict CPU, one cold inference per variant. Duration below excludes cold model loading. WER reference has 38 normalized tokens; **do not generalize to interviews or podcast WER**.

| Candidate | Cold load (s) | Inference (s) | RTF | WER | Native word timing |
| --- | ---: | ---: | ---: | ---: | --- |
| Nemotron 3.5 ASR 0.6B Q8_0 (transcribe.cpp 0.3) | 1.142 | **5.991** | **0.423** | **5/38 = 13.16%** | FAIL: 37 rows for 36 recognized words; extra empty word and final end 14.24s beyond 14.17s audio |
| deepdml Whisper large-v3-turbo CT2, int8 CPU, beam=5 | 6.493 | 11.865 | 0.837 | 7/38 = 18.42% | 36/36 non-empty native word rows, plausible, unverified against hand labels |
| same Whisper Turbo, beam=1 | 6.035 | 11.621 | 0.820 | 7/38 = 18.42% | same word count and bounds, but changes proper-noun spelling |
| Qwen3-ASR 1.7B Q8_0 (transcribe.cpp 0.3) | 3.302 | 24.926 | 1.759 | 5/38 = 13.16% | NONE by model design; existing AutoSubs CPU aligner needed |

The only French clip is **synthetic**, and even 5/38 still misrecognizes some proper nouns. No winner certified for real French podcasts yet. No production configuration changed as a result of this ranking.

Separate control experiment under 1.5 CPU / 3 GiB: transcribe.cpp **0.3.0 → 0.4.0** Nemotron Q8 reduced cold/warm inference from 12.914/11.596 s to 9.356/9.400 s on the same fixture with identical hypothesis. That CPU gain is **not** a Vulkan certification; official 0.4 Nemotron word timestamp test still emitted the extra empty overrun word.

## Verified word-for-word integration

- **transcribe.cpp Nemotron native** exposes word entries but not reliably complete/in-bounds; reject defective native words and use the existing CPU AutoSubs WhisperX aligner.
- **transcribe.cpp Qwen3-ASR** does not natively emit word times; recognize with Qwen first, align original text with WhisperX CPU once.
- **faster-whisper/Whisper Turbo CT2** returns native word rows with `word_timestamps=True`; validate and align only when invalid (feature opt-in).
- **Parakeet ONNX** currently only exposes text and json in this fork. No validated `verbose_json.words` path for its separate executor; Parakeet *GGUF*, if a supported transcribe.cpp model, uses transcribe.cpp route instead. Do not advertise universal native word timestamps on every engine.
- **AutoSubs** already runs `align_timeline` *after* transcription when its `alignment_enabled` setting is true. Avoid double WhisperX alignment: leave Cloud9-Speaches's opt-in bridge disabled for AutoSubs if AutoSubs itself aligns. The live AutoSubs setting has not been certified in this test.
- Existing `autosubs-aligner` `:8006` on CPU produced 37 valid reference-aligned tokens (0.260–13.950s) and 36 valid Nemotron-hypothesis tokens. Qwen hypothesis 36 words similarly aligned in **1.728s HTTP**, word confidence avg 0.882, without altering recognized text. Neither timing consistency nor alignment confidence proves millisecond-accurate boundaries; only manually annotated French reel/podcast clips can.

## Why GPU benchmarking is required

transcribe.cpp upstream **AMD Radeon RADV RENOIR 4750U** results (NOT Cloud9) show Qwen3-ASR 1.7B Q8 ~2.47s Vulkan vs 4.10s CPU on 11s JFK, and 9.49s Vulkan vs 15.63s CPU on 35s speech. Nemotron Q8 ~0.64s Vulkan vs 0.95s CPU on 11s JFK. These are hardware- and benchmark-specific and cannot be extrapolated as percentages for the **Radeon 780M**.

Sources:
- https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/qwen3-asr-1.7b.md
- https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/nemotron-3.5-asr-streaming-0.6b.md
- https://github.com/SYSTRAN/faster-whisper (GPU backend CTranslate2 is **CUDA/cuDNN**, not AMD Vulkan; cannot claim GPU Whisper Turbo performance using the existing CTranslate2 image)

### Candidates and quantization, not auto-downloaded

Actual local HF model snapshots: `handy-computer/Qwen3-ASR-1.7B-gguf` Q8 and `handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf` Q8, plus `deepdml/faster-whisper-large-v3-turbo-ct2`. At 2026-10-13 the cached Qwen and Nemotron revisions matched HF published revisions 2026-09-15 and Whisper deepdml cache matched its repo 2026-02-22.

Official GGUF variants:
- Qwen3 1.7B Q8 2.185 GB, Q6_K 1.693 GB, Q5_K_M 1.517 GB, Q4_K_M 1.320 GB. Upstream **English** test-clean WER Q8 1.62%, Q5 1.65%, Q4 1.81%; no matched French quantization result on Cloud9.
- Nemotron 0.6B Q8 751 MB, Q5 560 MB, Q4 496 MB. Prior upstream Q8 vs Q4 inference latencies were similar on AMD 4750U; smaller model is not necessarily faster than Q8.
- `Qwen3-ASR-0.6B` exists and is represented by the `stt-balanced` alias, but its model is **not currently cached** in Cloud9-Speaches. Do not claim it was benchmarked.

**Prioritize existing Q8, then selectively Q5/Q4 only if an actual Vulkan measurement indicates GPU memory pressure, throughput bottleneck or sufficient French WER.** With a nearly full host swap, never download extra weights merely for speculation.

## Reproducible GPU benchmark (host only, safely staged)

- `scripts/cloud9-host-gpu-preflight.py`: read-only host-only gate from existing Cloud9 Engine lab. Check GTT, host-wide D-state tasks, RAM, PSI, 200ms swap page movement, recent AMDGPU kernel errors. Exit **4 = REFUSE**.
- `scripts/cloud9-host-vulkan-ab.sh --check`: run the host gate, **no GPU job, no new Docker container**.
- `scripts/cloud9-host-vulkan-ab.sh --run nemotron-q8` or `--run qwen3-1.7b-q8`: explicit manual per-model run **only after clean host check**; one ephemeral CT410 Docker container with Vulkan render device, 2 CPUs, 5 GiB, no network, cached weights mounted read-only; 3 inference passes with model load and word row checks.
- Only if extra quantizations are explicitly cached: `--run qwen3-1.7b-q5`, `--run qwen3-1.7b-q4`, `--run nemotron-q5`, `--run nemotron-q4`.
- `scripts/cloud9_fr_cached_asr_cpu_ab.py` reproduces the CPU baseline, with beam 1/5 option for Whisper Turbo.

**Important**: the GPU is shared with production. Even after host preflight passes, choose an idle window, monitor CT410 production latency and GTT, and stop testing on any degradation. A passing preflight is a *necessary*, not sufficient, condition.

As of **09:28 CEST**: GPU preflight still REFUSES because Proxmox kernel tasks 2625774 (`drm_suballoc_new`) and 2683546 (`ttm_bo_evict_first`) remain in `D`, and GTT idle reports 11,427,979,264 bytes (~10.64 GiB). No new GPU workload was launched.

## Decision gate for AutoSubs

Compare **complete job**: upload/decode + STT load + STT cold/warm + alignment (if needed) + subtitle lines/ASS/SRT + render. Use at least a representative French Reel and podcast speech clips, annotate manual word boundaries to compute p50/p95 timing error and WER/CER, include overlapping speakers, accents, French hyphens, proper names, pauses, and known podcast intro/outro. Capture CPU%, GPU GTT/VRAM peaks, job seconds and errors. **Do not choose default models or enable duplicate alignment until those measurements exist.**
