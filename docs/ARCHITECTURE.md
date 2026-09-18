# Speaches Reloaded architecture

Speaches Reloaded keeps Speaches as the OpenAI-compatible API, UI and model-lifecycle layer. It adds transcribe.cpp as a second STT executor instead of replacing the existing faster-whisper, Kokoro or Piper paths.

## Runtime paths

~~~text
                        ┌─ faster-whisper ── Whisper STT
OpenAI-compatible API ──┼─ transcribe.cpp ── GGUF STT / streaming / diarization
                        ├─ Kokoro ────────── TTS
                        └─ Piper ─────────── TTS
~~~

The requested model determines the executor. Dynamic model loading and TTL-based unloading remain controlled by Speaches.

## Hardware profiles

- **CPU**: transcribe.cpp CPU + faster-whisper CPU.
- **Vulkan**: transcribe.cpp Vulkan on AMD or Intel; faster-whisper stays on CPU.
- **CUDA**: transcribe.cpp CUDA + faster-whisper CUDA on NVIDIA.
## Image design

All three release images use the same application source and pinned transcribe.cpp revision. Only the native backend and runtime base change. AMD and Intel share Vulkan because splitting them would duplicate the same backend and Mesa runtime.

Model weights are deliberately outside OCI images. The persistent Hugging Face cache survives upgrades, while scripts/models.sh provides lean, recommended and full download presets.

## Why heavy TTS engines are not bundled

Qwen3-TTS and Chatterbox add useful cloning and expressive features, but also add a large PyTorch/CUDA/ROCm dependency stack. Reloaded keeps the base speech server small and predictable; those engines belong in optional workers only when there is a concrete integration requirement.

## Upstream relationship

The repository retains Speaches Git history. Reloaded-specific commits stay above the pinned upstream base so upstream changes can be reviewed and rebased instead of copied into an unrelated codebase.
