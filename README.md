# Speaches Extended

OpenAI-compatible local STT/TTS server based on [Speaches](https://github.com/speaches-ai/speaches), extended with a native `transcribe.cpp` execution path and a portable Vulkan container.

> Unofficial community distribution. Not affiliated with the Speaches project.

## Why

Upstream Speaches is an OpenAI-compatible speech gateway built around faster-whisper, Kokoro and Piper. This distribution keeps that API/UI and adds a second STT engine: `transcribe.cpp`, which exposes many GGUF speech model families through native backends.

The Vulkan image targets Linux servers with AMD/Intel/NVIDIA graphics where a light native backend is preferable to a full PyTorch/ROCm stack.

## Highlights

- OpenAI-compatible transcription, translation, realtime and TTS endpoints.
- `faster-whisper` and `transcribe.cpp` behind the same API.
- Vulkan STT acceleration with dynamic model load/offload.
- Streaming-capable Nemotron/Parakeet-family support.
- Qwen3-ASR long-audio batching.
- `transcribe.cpp` Sortformer diarization path.
- Silero v6 / faster-whisper 1.2 compatibility fixes.
- Portable multi-stage Docker build with no Cloud9-specific artifacts.
- Curated aliases for fast/quality French and general STT/TTS use.
- Optional one-command model preloading.

## Quick start

```bash
cp .env.example .env
docker compose -f compose.vulkan.yaml up -d --build
curl http://localhost:8000/health
```

Optional curated model preload:

```bash
./scripts/preload-models.sh
```

The Hugging Face cache is stored in a Docker volume and survives upgrades.

## Default aliases

| Alias | Model |
|---|---|
| `stt-fast` / `fr-fast` | Nemotron 3.5 ASR Streaming 0.6B GGUF |
| `stt-quality` / `fr-quality` | Qwen3-ASR 1.7B GGUF |
| `stt-whisper` / `whisper-1` | faster-whisper large-v3-turbo |
| `stt-draft` | faster-whisper small |
| `tts-fast` | Piper French Tom |
| `tts-quality` | Kokoro 82M ONNX int8 |
