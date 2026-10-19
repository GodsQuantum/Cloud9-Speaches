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

A CI candidate requires Python imports, the genuine FastAPI routes,
`crispasr --help` starting successfully from the extracted bundle,
`/health` and `/v1/audio/alignment/health` HTTP 200 from the frozen process,
and an archive checksum.

**Not yet proven** by these smoke checks: microphone device integration,
model download/licensing, accurate word alignment, CUDA/Metal/Vulkan GPU
acceleration, long-audio performance, production Windows NSIS installation,
Apple Developer ID signing, notarization, Gatekeeper or Windows Authenticode.
These must be certified separately before a stable release.

The existing Cloud9 CT410 Vulkan production image is unchanged.
