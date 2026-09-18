# Architecture

Speaches Extended keeps the OpenAI-compatible Speaches API and adds a second STT execution path backed by `transcribe.cpp`.

## STT
- `faster-whisper`: compatibility fallback and Whisper-family production path.
- `transcribe.cpp` + Vulkan: GGUF models, AMD/Intel/NVIDIA Vulkan, streaming-capable model families, low dependency overhead.
- Dynamic model loading/offloading remains managed by Speaches.
- The same `/v1/audio/transcriptions` endpoint selects the backend from the requested model ID.

## TTS
- Kokoro ONNX: default quality/size balance.
- Piper: default low-latency/lightweight path.
- Heavier cloning/expressive engines such as Qwen3-TTS or Chatterbox are intentionally not bundled in the base image because they require a much larger PyTorch/ROCm/CUDA stack. They are candidates for optional workers/profiles, not base dependencies.

## Model distribution
Models are not committed to Git. `scripts/preload-models.sh` fills the persistent Hugging Face cache. This gives a reproducible plug-and-play deployment without forcing every image pull to include many gigabytes of model weights.
