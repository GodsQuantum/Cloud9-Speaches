<p align="center">
  <img src="https://raw.githubusercontent.com/GodsQuantum/Speaches-Reloaded/main/docs/assets/speaches-reloaded-logo.svg" width="138" alt="Speaches Reloaded 标志">
</p>

<h1 align="center">Speaches Reloaded</h1>

<p align="center">
  <strong>一个兼容 OpenAI 的语音 API。支持 CPU、Vulkan 和 CUDA。</strong><br>
  将 Speaches 与 transcribe.cpp 组合为适用于 NVIDIA、AMD、Intel 和纯 CPU 服务器的自托管 STT/TTS。
</p>

<p align="center">
  <img alt="Docker" src="https://img.shields.io/badge/runtime-Docker-2496ed">
  <img alt="GHCR" src="https://img.shields.io/badge/images-GHCR-54cd8a">
  <img alt="CPU Vulkan CUDA" src="https://img.shields.io/badge/backends-CPU%20%7C%20Vulkan%20%7C%20CUDA-9d8cf5">
</p>

<p align="center">🇬🇧 <a href="README.md">English</a> · 🇫🇷 <a href="README.fr.md">Français</a></p>

---

**Speaches Reloaded** 是 Speaches 的非官方社区发行版。它保留 Speaches 的 Web UI、模型生命周期管理和 OpenAI 兼容 API，同时加入 transcribe.cpp，让同一个服务可以运行现代 GGUF 语音模型。

## ✨ 主要特点

- **三种优化镜像** — 纯 CPU、AMD/Intel Vulkan、NVIDIA CUDA。
- **统一 API** — 现有 Speaches/OpenAI 客户端无需更改调用方式。
- **双 STT 引擎** — faster-whisper 与 transcribe.cpp。
- **TTS 保留** — 继续使用 Kokoro 和 Piper。
- **模型缓存持久化** — 更新容器时无需重新下载模型权重。
## 🖥️ 选择硬件配置

| 硬件 | 镜像 | Compose |
|---|---|---|
| 纯 CPU | ghcr.io/godsquantum/speaches-reloaded:latest-cpu | compose.cpu.yaml |
| AMD GPU / iGPU | ghcr.io/godsquantum/speaches-reloaded:latest-vulkan | compose.vulkan.yaml |
| Intel GPU / iGPU | ghcr.io/godsquantum/speaches-reloaded:latest-vulkan | compose.vulkan.yaml |
| NVIDIA GPU | ghcr.io/godsquantum/speaches-reloaded:latest-cuda | compose.cuda.yaml |

AMD 与 Intel 共用 Vulkan 镜像，因为两者使用相同的推理后端，无需维护重复镜像。

**即使机器装有 GPU，也始终可以只使用 CPU。** 如果希望完全释放 GPU，可在任何 x86-64 主机上直接使用 `compose.cpu.yaml`。已运行 Vulkan 镜像时，可设置 `TRANSCRIBE_BACKEND=cpu`；CUDA 镜像还可同时设置 `WHISPER_DEVICE=cpu`。如果希望 Docker 完全不映射或保留 GPU，CPU Compose 是最干净的选择。

## 🚀 安装

~~~bash
git clone https://github.com/GodsQuantum/speaches-reloaded.git
cd speaches-reloaded
cp .env.example .env
~~~

**CPU：** docker compose -f compose.cpu.yaml up -d

**AMD / Intel：**

~~~bash
RENDER_GID="$(stat -c '%g' /dev/dri/renderD128)"
sed -i "s/^RENDER_GID=.*/RENDER_GID=$RENDER_GID/" .env
docker compose -f compose.vulkan.yaml up -d
~~~

**NVIDIA：** docker compose -f compose.cuda.yaml up -d

NVIDIA 主机还需要安装 NVIDIA Container Toolkit。默认 Web UI 与 API 端口为 8000。
## 🧠 下载推荐模型

容器启动后运行：

~~~bash
./scripts/models.sh
~~~

脚本提供三个模型包：

| 模型包 | 用途 |
|---|---|
| lean | Nemotron 流式识别 + Kokoro + Piper |
| recommended | lean + Qwen3-ASR 0.6B + Whisper Large-v3-Turbo |
| full | recommended + Qwen3-ASR 1.7B + Parakeet v3 + Sortformer 说话人分离 |

Hugging Face 模型存放在持久化 Docker 卷 speaches-models 中，不会被打包进镜像，因此升级应用不会重复保存数 GB 权重。

## 🎙️ 简化模型别名

- stt-fast / fr-fast → Nemotron 3.5 ASR Streaming 0.6B
- stt-balanced → Qwen3-ASR 0.6B
- stt-quality / fr-quality → Qwen3-ASR 1.7B
- stt-whisper / whisper-1 → faster-whisper Large-v3-Turbo
- stt-europe → Parakeet TDT 0.6B v3
- tts-fast → Piper
- tts-quality → Kokoro 82M ONNX int8

## 许可证

Speaches Reloaded 是非官方社区发行版，保留 Speaches 的 Git 历史与 MIT 许可证。transcribe.cpp 作为外部依赖构建，并保留其自身许可证及第三方声明。
