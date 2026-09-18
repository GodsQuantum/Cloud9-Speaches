# Speaches Reloaded

Speaches Reloaded is an unofficial community distribution of Speaches with an additional native transcribe.cpp execution path.

It keeps the OpenAI-compatible API, UI, model lifecycle, faster-whisper, Kokoro and Piper paths from Speaches, while adding GGUF speech models through CPU, Vulkan or CUDA.

## What Reloaded adds

- CPU-only image for any x86-64 server.
- Vulkan image for AMD and Intel GPUs/iGPUs.
- CUDA image for NVIDIA GPUs.
- Runtime CPU fallback even when an accelerated image is installed.
- transcribe.cpp model families behind the same transcription API.
- Curated model aliases and an optional model download helper.
- Persistent Hugging Face model cache.

Start with [Installation](installation.md), then use the existing Speaches API documentation in this repository for endpoint details.

See [Architecture](ARCHITECTURE.md) for the Reloaded-specific execution paths.
