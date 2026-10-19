# Cloud9-Speaches native portable build — candidate

This packages the actual upstream-derived FastAPI/Gradio Speaches server with Python
3.12 inside PyInstaller **onedir**. It does not require Docker or a user-installed
Python runtime. The HTTP server binds only to `127.0.0.1:8005` by default.

## Runtime and provenance

- CrispASR **v0.8.43** is bundled as an external executable, including its
  original licenses and runtime libraries; original GitHub asset SHA256 is
  checked before extraction. Windows defaults to x86_64 CPU, Linux to
  CPU/x86_64-legacy or ARM64, macOS to Apple Silicon Metal-capable or Intel CPU.
- Authentication is not enabled by default; the service binds to loopback
  only. Do **not** expose a port on a public interface without authentication.
- FastConformer aligner models and Whisper STT models are **not included** in
  preliminary CI candidates. They must be acquired separately from verified
  upstream URLs with clear licenses, correct hashes and first-run state.
- Runtime files are in `%LOCALAPPDATA%/Cloud9-Speaches` (Windows),
  `~/Library/Application Support/Cloud9-Speaches` (macOS), or
  `~/.local/share/Cloud9-Speaches` (Linux).
- Default `CLOUD9_CTC_BACKEND=cpu` avoids assuming GPU availability. The
  bundled macOS Apple Silicon executable also has Metal capability, but
  certification requires a real Metal transcription/forced-alignment fixture.
- PyInstaller CPython build/install is performed natively on each OS; never
  cross-build. PyInstaller official requirements:
  https://pyinstaller.org/en/stable/requirements.html
- CrispASR upstream and checksums:
  https://github.com/CrispStrobe/CrispASR/releases/tag/v0.8.43
  https://github.com/CrispStrobe/CrispASR/blob/main/docs/install.md

## Test gates

Windows NSIS setup installs a real PyInstaller binary, tests HTTP then uninstalls it, preserving per-user settings. The portable ZIP remains available.

A CI candidate requires Python imports, the genuine FastAPI routes,
`crispasr --help` starting successfully from the extracted bundle,
`/health` and `/v1/audio/alignment/health` HTTP 200 from the frozen process,
and an archive checksum.

**Not yet proven** by these smoke checks: microphone device integration,
model download/licensing, accurate word alignment, CUDA/Metal/Vulkan GPU
acceleration, long-audio performance, Windows Authenticode signing,
Apple Developer ID signing, notarization, Gatekeeper or Windows Authenticode.
These must be certified separately before a stable release.

The existing Cloud9 CT410 Vulkan production image is unchanged.

## Install the verified French model pack (opt-in)

After installing, the model bootstrap downloads 762 MB:
the official FastConformer French Q4_K GGUF (69,651,168 bytes, CC-BY-4.0)
and transcribe.cpp Whisper Large-v3-Turbo Q6_K (692,536,896 bytes, MIT).
To install through the bundled native executable, run:

    Cloud9-Speaches --install-default-models
    Cloud9-Speaches --install-default-models --model-group aligner

Immutable Hugging Face commits, SHA256 digests and byte sizes are checked
before declaring installation complete. Models reside only in application
per-user data, never in the application bundle. The native Hugging Face
cache is used for the speech recognizer. No compatibility symlinks are made.
Model bootstrap is never executed silently from the CI smoke checks.

After the model pack is verified, Cloud9-Speaches will discover Q6_K Turbo
and French FastConformer in the one-click AutoSubs/ScribeWatch pairing UI.
Real transcription and alignment still require an acoustic sample test.

## Native transcribe.cpp 0.4.0 providers (CPU/Vulkan/Metal)

The portable app now depends on official transcribe-cpp-native 0.4.0 Python
wheels, not system-level user-installed drivers or Docker. PyInstaller
collects both the native modules and entry-point metadata so that the
real native library can load. The native install checker requires a
CPU backend; Vulkan and Metal are optional and need hardware evaluation.
Linux x64 and Windows x64 use Vulkan where a loader and driver are present.
Apple Silicon uses Metal plus CPU; Intel Mac uses CPU. CUDA remains an
optional provider and cannot be certified until tested with NVIDIA hardware.
These are backend *capabilities*; an actual GGUF must be downloaded separately
and a real audio fixture processed before certifying inference performance.
