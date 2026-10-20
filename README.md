<p align="center">
  <img src="https://raw.githubusercontent.com/GodsQuantum/Cloud9-Speaches/main/docs/assets/speaches-reloaded-logo.svg" width="138" alt="Cloud9-Speaches logo">
</p>

<h1 align="center">Cloud9-Speaches</h1>

<p align="center">
  <strong>One OpenAI-compatible speech API. CPU, Vulkan or CUDA.</strong><br>
  Cloud9-Speaches: Speaches + transcribe.cpp, packaged for fast self-hosted STT/TTS on NVIDIA, AMD, Intel and CPU-only servers.
</p>

<p align="center">
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-3dd7cf"></a>
  <img alt="Docker" src="https://img.shields.io/badge/runtime-Docker-2496ed">
  <img alt="GHCR" src="https://img.shields.io/badge/images-GHCR-54cd8a">
  <img alt="CPU Vulkan CUDA" src="https://img.shields.io/badge/backends-CPU%20%7C%20Vulkan%20%7C%20CUDA-9d8cf5">
  <a href="https://github.com/GodsQuantum/Cloud9-Speaches/actions/workflows/release-images.yml"><img alt="Images" src="https://github.com/GodsQuantum/Cloud9-Speaches/actions/workflows/release-images.yml/badge.svg"></a>
</p>

<p align="center">🇫🇷 <a href="README.fr.md">Français</a> · 🇨🇳 <a href="README.zh-CN.md">简体中文</a></p>

---

**Cloud9-Speaches** (formerly Speaches Reloaded) is a community distribution of [Speaches](https://github.com/speaches-ai/speaches). It keeps the Speaches UI and OpenAI-compatible API, then adds a native [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) STT path so the same server can use modern GGUF speech models without forcing every machine into one inference stack.

The goal is simple: clone the repo, choose the Compose file matching the hardware, and keep the same OpenAI-compatible /v1 speech API.

> **Repository transition:** the product/canonical name is now **Cloud9-Speaches**. The GitHub repository has been renamed to [GodsQuantum/Cloud9-Speaches](https://github.com/GodsQuantum/Cloud9-Speaches), preserving commit history and GitHub redirects. Existing GHCR image tags remain `ghcr.io/godsquantum/speaches-reloaded` for deployment compatibility; do not change image URLs until new package tags are published and validated.
## ✨ Why Cloud9-Speaches?

- **Three optimized images** — CPU-only, Vulkan for AMD/Intel, CUDA for NVIDIA.
- **One API** — keep Speaches-compatible clients, integrations and model discovery.
- **Three STT executors** — transcribe.cpp for GGUF (Nemotron/Qwen3-ASR), faster-whisper for Whisper CT2, plus the optional ONNX Parakeet path.
- **Modern model families** — Nemotron, Qwen3-ASR, Parakeet, Whisper and more through transcribe.cpp.
- **Streaming + diarization paths** — model capabilities remain exposed behind the same server.
- **TTS included** — Speaches keeps its Kokoro and Piper paths.
- **Persistent model cache** — container upgrades do not redownload every model.
- **No host-specific artifacts** — published images build from upstream sources and pinned commits.
- **Multi-node ready** — replicate workers on several machines and validate them with `scripts/cluster-doctor.sh`; ordinary STT/TTS jobs use request-level load balancing while realtime connections remain pinned to one worker. See [docs/distributed.md](docs/distributed.md).

## 🎯 Verified word timestamps and Radeon benchmarking

**Word-for-word captions:** Qwen3-ASR (GGUF/transcribe.cpp) needs a forced aligner because it has no native word times. Nemotron's native word outputs must be validated (an extra empty word beyond the audio end was observed). Whisper Turbo CT2 supports native word timing; all three can use the **existing AutoSubs CPU WhisperX aligner**, enabled explicitly, for corrected timing. The bridge is **opt-in**; avoid performing alignment twice when AutoSubs already does it. See [word-level timing contract](docs/cloud9-autosubs-word-timestamps-20261013.md).

**CPU vs AMD Vulkan:** Real Ryzen 7 8845HS CPU results and the strictly gated Vulkan benchmark harness are recorded in [Cloud9 Radeon 780M ASR benchmarks](docs/cloud9-amd-780m-asr-cpu-vulkan-benchmarks-20261013.md). **No Vulkan gain is claimed without a healthy GPU and a real benchmark.** The separate faster-whisper/CTranslate2 backend does not gain AMD Vulkan acceleration by selecting transcribe.cpp's Vulkan setting.

**Release policy:** Ordinary merges to `main` do not auto-publish the existing GHCR `latest-*` images. Tagged releases or explicit manual dispatch build them after contract and quality gates.

## 🖥️ Pick your hardware

| Hardware | Image | Compose | Main acceleration |
|---|---|---|---|
| CPU only | ghcr.io/godsquantum/speaches-reloaded:latest-cpu | compose.cpu.yaml | transcribe.cpp CPU + faster-whisper CPU |
| AMD GPU / iGPU | ghcr.io/godsquantum/speaches-reloaded:latest-vulkan | compose.vulkan.yaml | transcribe.cpp Vulkan |
| Intel GPU / iGPU | ghcr.io/godsquantum/speaches-reloaded:latest-vulkan | compose.vulkan.yaml | transcribe.cpp Vulkan |
| NVIDIA GPU | ghcr.io/godsquantum/speaches-reloaded:latest-cuda | compose.cuda.yaml | transcribe.cpp CUDA + faster-whisper CUDA |

AMD and Intel intentionally share one Vulkan image. Maintaining separate images would add duplication without changing the inference backend.

**CPU is always an option, even when a GPU is installed.** Use `compose.cpu.yaml` on any x86-64 machine when you want the GPU completely free. If you already run the Vulkan image, set `TRANSCRIBE_BACKEND=cpu` to move transcribe.cpp to CPU. With the CUDA image, set both `TRANSCRIBE_BACKEND=cpu` and `WHISPER_DEVICE=cpu`; use the CPU Compose when you want Docker to stop reserving/mapping the GPU entirely.
## 🚀 Quick start

Requirements: Docker Engine + Docker Compose v2. NVIDIA users also need the NVIDIA Container Toolkit.

~~~bash
git clone https://github.com/GodsQuantum/Cloud9-Speaches.git
cd Cloud9-Speaches
cp .env.example .env
~~~

**CPU only**

~~~bash
docker compose -f compose.cpu.yaml up -d
~~~

**AMD or Intel Vulkan**

~~~bash
RENDER_GID="$(stat -c '%g' /dev/dri/renderD128)"
sed -i "s/^RENDER_GID=.*/RENDER_GID=$RENDER_GID/" .env
docker compose -f compose.vulkan.yaml up -d
~~~

**NVIDIA CUDA**

~~~bash
docker compose -f compose.cuda.yaml up -d
~~~

Open http://SERVER:8000 or test the API with curl against http://SERVER:8000/health.
## 🧠 Download recommended models

Start the container first, then run:

~~~bash
bash ./scripts/models.sh
~~~

The helper offers three packs:

| Pack | Intended use |
|---|---|
| lean | Nemotron streaming + Kokoro + Piper |
| recommended | lean + Qwen3-ASR 0.6B + Whisper Large-v3-Turbo |
| full | recommended + Qwen3-ASR 1.7B + Parakeet v3 + Sortformer diarization |

Run bash ./scripts/models.sh recommended non-interactively, or pass exact Hugging Face repositories with bash ./scripts/models.sh custom MODEL_ID....

Models live in the persistent speaches-models Docker volume. They are not baked into the image, so upgrades do not duplicate gigabytes of weights.

## 🎙️ Friendly model aliases

| Alias | Default model |
|---|---|
| stt-fast / fr-fast | Nemotron 3.5 ASR Streaming 0.6B |
| stt-balanced | Qwen3-ASR 0.6B |
| stt-quality / fr-quality | Qwen3-ASR 1.7B |
| stt-whisper / whisper-1 | faster-whisper Large-v3-Turbo |
| stt-europe | Parakeet TDT 0.6B v3 |
| tts-fast | Piper fr_FR tom medium |
| tts-quality | Kokoro 82M ONNX int8 |
## ⚙️ Architecture

~~~text
OpenAI-compatible clients
          │
          ▼
  Cloud9-Speaches
    │     │      │
    │     │      ├── Kokoro / Piper ──► TTS
    │     ├──────── faster-whisper ───► Whisper STT
    └────────────── transcribe.cpp ───► GGUF STT / streaming / diarization
                       │
                CPU / Vulkan / CUDA
~~~

Speaches remains the API, UI and model lifecycle layer. Reloaded adds transcribe.cpp as another executor rather than replacing the mature faster-whisper, Kokoro or Piper paths.

See docs/ARCHITECTURE.md for the exact split.

## 📦 Published images

- ghcr.io/godsquantum/speaches-reloaded:latest-cpu
- ghcr.io/godsquantum/speaches-reloaded:latest-vulkan
- ghcr.io/godsquantum/speaches-reloaded:latest-cuda

The Dockerfiles pin the transcribe.cpp revision used for a release. Model weights keep their own upstream licenses and are downloaded separately.

## 🧪 Development

~~~bash
python tests/reloaded_release_test.py
git diff --check
docker build -f Dockerfile --build-arg TRANSCRIBE_BACKEND=cpu -t speaches-reloaded:dev .
~~~
## Credits & license

Cloud9-Speaches is an unofficial community distribution built on Speaches and transcribe.cpp. It is not affiliated with or endorsed by the upstream projects.

The Speaches code and this distribution retain the MIT license and upstream Git history. transcribe.cpp is built as an external dependency and retains its own license and third-party notices.

Thanks to the Speaches and transcribe.cpp maintainers for the foundations Reloaded packages together.

## Automatic Linux Docker CPU/Vulkan/CUDA hardware advice

**Downloads:** [Native CPU candidates for Linux x64/ARM64, Windows x64 and macOS Intel/Apple Silicon](https://github.com/GodsQuantum/Cloud9-Speaches/releases/tag/desktop-v0.1.0-rc.1). Native checks include a real CPU FastConformer model; **Windows/macOS packages have no official signing or Apple Developer ID notarization, and GPU acceleration is not certified on every client device**.

The Linux host doctor is **read-only**: choose CUDA only if nvidia-smi works, Vulkan only with an accessible render node AND working vulkaninfo; otherwise CPU. It neither changes Compose nor starts services:

~~~bash
python3 scripts/auto-profile.py
python3 scripts/auto-profile.py --env
# Review the suggested Compose file before using it:
docker compose -f compose.vulkan.yaml config
~~~

On Ryzen 7 8845HS with Radeon 780M, test transcribe.cpp GGUF via Vulkan, while faster-whisper/CTranslate2 remains CPU with WHISPER_DEVICE=cpu and around 4 threads. NVIDIA additionally requires the Container Toolkit and a real CUDA inference test. On limited hardware, start CPU-only and lean models. Benchmark Nemotron, Qwen3-ASR, Whisper and French FastConformer Q4 on identical representative audio, measuring quality, memory, word alignment and actual GPU utilization. **Device detection is not a performance benchmark or a substitute for a real smoke test.**
