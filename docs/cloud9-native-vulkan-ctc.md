# Cloud9-Speaches: embedded Vulkan forced alignment

## Architecture (17 October 2026)

Cloud9-Speaches CT410 / TCP 8005 runs transcription and CrispASR CTC forced
alignment INSIDE the app. A dedicated Gradio Acoustic Alignment tab and STT
advanced controls expose manual model selection.

Native API:
- GET /v1/audio/alignment/models: FR/EN/ZH defaults and available models.
- GET /v1/audio/alignment/health: Vulkan backend and registry.
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

Image: cloud9/speaches:native-ctc-validated-20261017.
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
AutoSubs 3.6.2 now uses Speaches :8005/align: real Reel ready,
128 subtitle blocks and 402 timed words. ScribeWatch direct Speaches
provider and OmniRoute provider tested.

Strict mode rejects absent longform ASR offsets, lexical divergence,
non-finite timestamps, and incomplete words. Never synthesize acoustic
timestamps or confidence. Mathematically perfect phonetic boundaries
are not claimed without manual waveform annotation.

## Rollback

Retained standalone CrispASR compose:
 /home/arezki/Stacks/autosubs-aligner/compose.vulkan.yaml (CT410).
Previous Speaches Docker tag: cloud9/speaches:fastconformer-20261017.
Restore the old HTTP URL :8010/align only if native regression detected,
keeping original mounts/cgroups and no changes to host NFS.

Canonical CTC source is src/speaches/executors/cloud9_ctc.py;
vulkan-aligner/server.py is retained for standalone recovery. Keep
the two copies byte-identical whenever core behavior changes.

## Legacy port and CT410 kernel key quota

The original standalone CTC endpoint on port 8010 is retained as a known-good
compatibility/rollback path, but is NO LONGER called by either Speaches or
AutoSubs production. Native :8005 is the authoritative integrated endpoint.
Do not remove or recreate the original container without checking all clients.

CT410 has near-exhausted Linux kernel root-user keyrings (~19,868 / 20,000
bytes on this verification), which caused a disposable docker run to fail
with "unable to create session key: disk quota exceeded". No host key quotas
or kernel settings were changed. Our test canary was removed. Avoid extra
container recreations until keyring quota is addressed separately.
