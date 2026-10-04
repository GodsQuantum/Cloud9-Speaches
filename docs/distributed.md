# Multi-node speech

Cloud9-Speaches scales across machines by **routing complete STT/TTS jobs to replicated workers**. Do not split one
audio transcription across machines at the engine level unless a model explicitly supports chunking with a
correct merge policy.

## Cloud9 + Celestra

Install the same release and desired model pack on both nodes, then verify:

`SPEACHES_CLUSTER_NODES=http://NODE1:8000,http://NODE2:8000 bash ./scripts/cluster-doctor.sh`

A health-aware reverse proxy can then distribute ordinary transcription and speech requests. Prefer
least-connections because audio jobs have highly variable durations.

## Realtime and streaming

A realtime WebSocket/SSE connection must stay on the same worker for its lifetime. Configure the reverse proxy for
WebSocket upgrades and connection affinity; new sessions may be sent to any healthy node.

## Backend policy

Each worker chooses its own best backend. On Cloud9 780M the validated production path remains Vulkan+CPU through
transcribe.cpp; a future Celestra worker may use Vulkan or ROCm only after its own WER/RTF/thermal gate. The API
surface stays identical, so heterogeneous worker hardware is acceptable when model/output compatibility is proven.

A worker failure should retry/restart the whole non-streaming request. Do not concatenate outputs from two engines
that used different segmentation state.
