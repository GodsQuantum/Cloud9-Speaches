# Cloud9-Speaches: embedded CPU/Vulkan forced alignment

## Architecture (17 October 2026)

Cloud9-Speaches CT410 / TCP 8005 runs transcription and CrispASR CTC forced
alignment INSIDE the app. A dedicated Gradio Acoustic Alignment tab and STT
advanced controls expose manual model selection.

Native API:
- GET /v1/audio/alignment/models: FR/EN/ZH defaults and available models.
- GET /v1/audio/alignment/health: configured CTC backend and registry (not a driver probe).
- POST /v1/audio/alignment/align (also /align for AutoSubs):
  PCM16 16 kHz mono WAV, transcript, segments_json, language, model.
- POST /v1/audio/transcriptions: standard Speaches API;
  response_format=verbose_json, timestamp_granularities[]=word and optional
  alignment_model=fastconformer-fr-q4 (or auto) produce acoustic word timings.

FR model: fastconformer-fr-q4
EN model: fastconformer-en-q4
ZH model: wav2vec2-zh-q4 (Han-character acoustic units)
Other supported languages: canary-ctc-multilingual-q4.

Weights remain in CT410 shared /srv/ai-models (read-only bind /ai-models).

## Native Speaches production settings

Image: cloud9/speaches:native-ctc-20261018-certified.
Previous validated image: cloud9/speaches:native-ctc-validated-20261017.
Preserve cgroup_parent: cloud9-protected.slice and existing memory limits.
CLOUD9_TRANSCRIBE_CPP_PREFERRED_QUANT=Q6_K
CLOUD9_NATIVE_CTC_ENABLED=1
CLOUD9_WORD_ALIGNMENT_URL=internal
CLOUD9_WORD_ALIGNMENT_STRATEGY=forced
CLOUD9_WORD_ALIGNMENT_STRICT=1
ALIGN_MODEL_ROOT=/ai-models/audio/asr/aligners
CRISPASR_BIN=/opt/crispasr/bin/crispasr
bind /srv/ai-models:/ai-models:ro, container only; no host mount changes.

Build recipe: deploy/cloud9/Dockerfile.native-ctc uses the retained,
qualified local images cloud9/speaches:fastconformer-20261017 and
cloud9/vulkan-ctc-aligner:20261017-zh. CrispASR Vulkan needs glibc 2.43;
Speaches Ubuntu 24.04 uses glibc 2.39. Newer glibc loader/libs reside
ONLY in /opt/ctc-glibc and are invoked for the CrispASR subprocess.
Never overwrite global libc or global LD_LIBRARY_PATH.
langdetect==1.0.9 is pinned for manual auto-language detection.

## Verified

Production Speaches CTC health 200. Real FR 104.960s Reel: 402 measured
CTC words, 3 acoustic windows, 2.45s. ZH sample: 65 Han units.
Canary full Turbo Q6_K Vulkan STT to native CTC: 22 EN word timestamps.
AutoSubs production uses Speaches :8005/align: real Reel ready,
128 subtitle blocks and 402 timed words. ScribeWatch direct Speaches
provider and OmniRoute provider tested.

Strict mode rejects absent longform ASR offsets, lexical divergence,
non-finite timestamps, and incomplete words. Never synthesize acoustic
timestamps or confidence. Mathematically perfect phonetic boundaries
are not claimed without manual waveform annotation.

## Rollback

**Production rollback (CT410, no mount change):**
- Compose: `/srv/lxc/ia-compute/compose/speaches/compose.yaml`.
- Current image: `cloud9/speaches:native-ctc-20261018-certified`.
- Previous tested image retained: `cloud9/speaches:native-ctc-validated-20261017`.
- In that Compose file replace ONLY the current `image:` value with the
  previous image tag. Run:
  `docker compose -f /srv/lxc/ia-compute/compose/speaches/compose.yaml up -d --no-deps --force-recreate speaches`
  inside CT410. Verify `http://127.0.0.1:8005/health` and
  `http://127.0.0.1:8005/v1/audio/alignment/health` return HTTP 200.
  Keep `cgroup_parent: cloud9-protected.slice`, all data binds and GPU access.
- A separate legacy CrispASR compose survives in CT410 at
  `/home/arezki/Stacks/autosubs-aligner/compose.vulkan.yaml`, but is STOPPED
  and its port 8010 is not currently active. Reactivate only for a deliberate
  fallback, never as an implicit dependency or without checking the client URL.
- Base image `cloud9/speaches:fastconformer-20261017` is retained for rebuilds.
Never change host NFS configuration as part of rollback.

Canonical CTC source is src/speaches/executors/cloud9_ctc.py;
vulkan-aligner/server.py is retained for standalone recovery. Keep
the two copies byte-identical whenever core behavior changes.

## Legacy port and CT410 kernel key quota

The old standalone CTC container is STOPPED; only its Compose recipe
and base image remain for rollback. Native port :8005 is the active API.
AutoSubs production alignment URL is `http://192.168.1.219:8005/align`.

CT410 unprivileged root mapped to host UID 100000 previously hit the kernel
keyring byte quota (~19,828 / 20,000), causing a disposable Docker run to
fail. The narrow host sysctl `kernel.keys.maxbytes=131072` was applied
and persisted in `/etc/sysctl.d/70-cloud9-lxc-docker-keyring.conf`.
On 18 October the CT410 root keyring read ~19,828 / 131,072 bytes and
new Docker canaries started successfully. Do not indiscriminately revoke
running session keys, edit NFS mounts or raise other unrelated sysctls.


## Upstream build provenance and cleanup

CrispASR Vulkan upstream source:
https://github.com/CrispStrobe/CrispASR.git
Pinned source commit: 369029c91d90a61d8346b8bbdafe426bfdde76d1
Qualified local binary SHA256:
3cf1321ebb417dfbee5470a9629719529e83a429b0e378fca41ee857a03326b9

After release, the local upstream checkout and published Docker archive are
disposable build products. Retain Docker image cloud9/vulkan-ctc-aligner:20261017-zh
and vulkan-aligner/docker/runtime/ (43 MiB local binary staging) to rebuild
or roll back the CTC backend without downloads. They are ignored by Git
and are not user data. Before updating, clone the pinned upstream revision,
rebuild and recertify Vulkan binaries.


## Cross-platform backend wiring (18 October 2026)

Source now accepts `CLOUD9_CTC_BACKEND=auto|cpu|cuda|vulkan|metal`
(for CrispASR CLI) and `TRANSCRIBE_BACKEND=auto|cpu|cuda|vulkan|metal`
(for transcribe.cpp), with portable temporary directories and a configurable
`CRISPASR_BIN` absolute path. Cloud9 currently runs Vulkan only; the default
CTC backend remains Vulkan specifically to avoid changing the qualified
production path. On native Windows/macOS/Linux installs, the launcher must set
both variables according to the hardware and its selected binary.

Important: backend *selection* is not backend *certification*. The installer
must ship suitable precompiled per-OS transcribe.cpp and CrispASR binaries,
check compiled backend and driver availability, download model weights and
verify actual inference/CTC timestamps. The health field `backend` reports
configured preference, not measured device use. A CPU build must be present
when planning a CPU fallback; upstream Vulkan-only distributions do not
necessarily fall back when no Vulkan driver is installed. Metal cannot be
compiled by the Linux Dockerfile, which remains Linux-specific; native macOS
bundling must use the upstream macOS build. CUDA must be built and tested on
NVIDIA hardware. Do not claim platform parity from this configuration patch.


## Final production qualification (18 October 2026)

Sources: PR #6 and PR #7 merged into Cloud9-Speaches `main` at commit
`598f507faae6b8fb322d480aad33e8aaca4af2b9`. Native CTC Python API
contract and strict alignment tests passed on GitHub-hosted Linux, Windows
and macOS runners. Ruff, formatter, Pyrefly and full Pytest checks all passed.

On CT410 Radeon 780M Vulkan, same 104.960-s French Reel:
Turbo GGUF Q6_K transcription fed real ASR segments to CrispASR
FastConformer FR Q4, 402/402 measured lexical words, 3 alignment windows,
last end 104.50 s, no interpolation. CPU forced alignment took 6.752 s;
Vulkan took 2.479 s. Short 12-s CPU/Vulkan tests also passed.
The image candidate was rebuilt with the complete source overlay;
the isolated Vulkan canary completed the integrated OpenAI-compatible
`/v1/audio/transcriptions` request (HTTP 200, 28 timestamped words).
Production image was then promoted while retaining the previous image.
Production tested again: HTTP 200, 28 timestamped words, 4.875 s total.

AutoSubs live production test after promotion: source Reel 104.96 s,
job `ready`, 128 subtitle blocks, no errors; test job deleted (HTTP 404).
ScribeWatch live test using its direct Cloud9-Speaches provider and Turbo Q6_K:
job `done`, Markdown exported; test job deleted (HTTP 404). Both original
media and mounts preserved. The AutoSubs **published Linux AppImage** is
`v3.6.3` (x86_64 and aarch64 plus SHA256). Its currently running Cloud9
Docker backend may still report `v3.6.2`; it already includes the strict
FastConformer change but should be version-aligned on its own release cycle.

CUDA and Metal binaries / real hardware throughput are **not certified**;
the Linux/Vulkan Docker image is NOT an OS-universal installer. CPU and Vulkan
CrispASR are verified on Cloud9 x86_64, and multi-OS Python contracts are
verified in CI. For native macOS/Windows delivery, separately bundle suitable
upstream transcribe.cpp and CrispASR binaries, audio dependencies, manifests,
models, signatures and native GPU tests.
