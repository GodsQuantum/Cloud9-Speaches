# Cloud9-Speaches: transcribe.cpp 0.4.0 CPU hardware candidate (2026-10-13)

**Candidate only. GPU/Vulkan NOT certified, no image promotion or production update.**

## Why version 0.4.0?

Official upstream [transcribe.cpp v0.4.0](https://github.com/handy-computer/transcribe.cpp/releases/tag/v0.4.0), pinned to `f1dcd15858c4bbba87ca97862b4e448b2cf15b8f`.

The published Linux CPU+Vulkan native asset (22,546,161 bytes) was verified against release SHA256 `f229adb9018bb7e33042137f92942022a5d76bfc5e16efd7a11cc9b115e86ea2`. Both the native library and Python bindings were checked out from exactly the same v0.4.0 commit.

**ABI correctness:** native 0.4 + bindings 0.3 raised an undefined-symbol exception (`transcribe_sortformer_stream_ext_init`). The paired 0.4 binaries/bindings imported successfully. This Dockerfile checks out the single pinned upstream ref and builds/installs native and Python bindings from that same source.

## Physical CPU-only A/B

On Cloud9 Ryzen 7 8845HS in CT410, ephemeral Docker tests, `--network none --cpus 1.5 --memory 3g --memory-swap 3g --read-only`, no mapped `/dev/dri`; both engines explicitly selected `cpu` strict backend.

GGUF `handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf` Q8_0 (751,094,240 bytes), French synthetic audio `bench/fixtures/fr_regression_v1.wav`, length 14.17 s. Two consecutive transcriptions after cold loading each version.

| Metric | 0.3.0 | 0.4.0 |
|---|---:|---:|
| Model load (s) | 1.88774 | **1.35857** |
| First STT run (s) | 12.91365 | **9.35553** |
| Second STT run (s) | 11.59560 | **9.40003** |
| Total test time (s) | 26.40236 | **20.11675** |
| Normalized word error rate (WER) | 5/38 = 13.16% | same |
| Predicted text | identical | identical |

First-run latency -27.55%, second-run -18.93%. Very short sample with only one synthetic French utterance: **not a statistically sound universal performance claim**. Both predicted identical text; proper names were imperfect in both. Results are not directly comparable to the separate live `faster-whisper-small` HTTP benchmark because that ran without the same CPU cgroup cap.

## Critical GPU exclusion and remaining release gates

The 13 October AMDGPU driver is **still wedged**: host GPU tasks in `D` and unbounded GTT; full GPU test is unsafe. We did not reset AMDGPU, kill unrelated LXC work, preempt other engines, change mounts or reboot.

Before merging candidate into the production default and rebuilding the published `latest-vulkan` image:

1. Clean GPU confirmation, no stuck `D` task and healthy preflight GTT/PSI;
2. Equal-model 0.3.0 vs 0.4.0 cold/warm Vulkan STT on real device, memory accounting, unload/reload repeated, no DeviceLost;
3. French WER and RTF with multiple clips including podcast speech, not only synthetic;
4. Streaming, diarization, timestamps, API compatibility, CPU fallback regression;
5. CPU, Vulkan and CUDA image builds and CI contract/quality gates, stable GHCR tags;
6. Production promotion, with rollback, only after criteria pass.

GitHub repo now `GodsQuantum/Cloud9-Speaches`; **old GHCR package path remains `ghcr.io/godsquantum/speaches-reloaded`** intentionally to preserve existing deployments.

Canonical report: `Moteurs IA/Sources/Worklogs/2026-10-13_0230_CLOUD9_SPEACHES_CPU_AB_RENAME.md`. Machine-readable source: `Moteurs IA/Sources/Records/SPEECH_BENCHMARK_LEDGER.jsonl`.
