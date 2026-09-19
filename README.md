<p align="center">
  <img src="https://raw.githubusercontent.com/GodsQuantum/Speaches-Reloaded/main/docs/assets/speaches-reloaded-logo.svg" width="138" alt="Speaches Reloaded logo">
</p>

<h1 align="center">Speaches Reloaded</h1>

<p align="center">
  <strong>One OpenAI-compatible speech API. CPU, Vulkan or CUDA.</strong><br>
  Speaches + transcribe.cpp, packaged for fast self-hosted STT/TTS on NVIDIA, AMD, Intel and CPU-only servers.
</p>

<p align="center">
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-3dd7cf"></a>
  <img alt="Docker" src="https://img.shields.io/badge/runtime-Docker-2496ed">
  <img alt="GHCR" src="https://img.shields.io/badge/images-GHCR-54cd8a">
  <img alt="CPU Vulkan CUDA" src="https://img.shields.io/badge/backends-CPU%20%7C%20Vulkan%20%7C%20CUDA-9d8cf5">
  <a href="https://github.com/GodsQuantum/speaches-reloaded/actions/workflows/release-images.yml"><img alt="Images" src="https://github.com/GodsQuantum/speaches-reloaded/actions/workflows/release-images.yml/badge.svg"></a>
</p>

<p align="center">🇫🇷 <a href="README.fr.md">Français</a> · 🇨🇳 <a href="README.zh-CN.md">简体中文</a></p>

---

**Speaches Reloaded** is a community distribution of [Speaches](https://github.com/speaches-ai/speaches). It keeps the Speaches UI and OpenAI-compatible API, then adds a native [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) STT path so the same server can use modern GGUF speech models without forcing every machine into one inference stack.

The goal is simple: clone the repo, choose the Compose file matching the hardware, and keep the same OpenAI-compatible /v1 speech API.
## ✨ Why Reloaded?

- **Three optimized images** — CPU-only, Vulkan for AMD/Intel, CUDA for NVIDIA.
- **One API** — keep Speaches-compatible clients, integrations and model discovery.
- **Two STT engines** — faster-whisper for Whisper compatibility plus transcribe.cpp for GGUF speech models.
- **Modern model families** — Nemotron, Qwen3-ASR, Parakeet, Whisper and more through transcribe.cpp.
- **Streaming + diarization paths** — model capabilities remain exposed behind the same server.
- **TTS included** — Speaches keeps its Kokoro and Piper paths.
- **Persistent model cache** — container upgrades do not redownload every model.
- **No host-specific artifacts** — published images build from upstream sources and pinned commits.

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
git clone https://github.com/GodsQuantum/speaches-reloaded.git
cd speaches-reloaded
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
./scripts/models.sh
~~~

The helper offers three packs:

| Pack | Intended use |
|---|---|
| lean | Nemotron streaming + Kokoro + Piper |
| recommended | lean + Qwen3-ASR 0.6B + Whisper Large-v3-Turbo |
| full | recommended + Qwen3-ASR 1.7B + Parakeet v3 + Sortformer diarization |

Run ./scripts/models.sh recommended non-interactively, or pass exact Hugging Face repositories with ./scripts/models.sh custom MODEL_ID....

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
  Speaches Reloaded
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

Speaches Reloaded is an unofficial community distribution built on Speaches and transcribe.cpp. It is not affiliated with or endorsed by the upstream projects.

The Speaches code and this distribution retain the MIT license and upstream Git history. transcribe.cpp is built as an external dependency and retains its own license and third-party notices.

Thanks to the Speaches and transcribe.cpp maintainers for the foundations Reloaded packages together.
