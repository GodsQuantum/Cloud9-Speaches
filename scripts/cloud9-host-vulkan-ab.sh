#!/usr/bin/env bash
# Cloud9 HOST ONLY. Refuse unsafe GPU. Do not stop or restart services.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
model="${2:-nemotron-q8}"
mode="${1:---check}"

# Gate FIRST; must not execute pct/docker/copy if host GPU is wedged.
python3 -B "$ROOT/scripts/cloud9-host-gpu-preflight.py"

if [[ "$mode" == "--check" ]]; then
  echo "SAFE_HOST_PREFLIGHT_ONLY: no model run"
  exit 0
fi
if [[ "$mode" != "--run" || "$EUID" -ne 0 ]]; then
  echo "Usage (on Proxmox host, root): $0 --check | --run <model>" >&2
  exit 2
fi
case "$model" in
  nemotron-q8|nemotron-q5|nemotron-q4|qwen3-1.7b-q8|qwen3-1.7b-q5|qwen3-1.7b-q4) ;;
  *) echo "Unrecognized model; no implicit download" >&2; exit 2 ;;
esac

# Explicit one-test concurrency restriction; no stopping other containers.
exec 8>/run/lock/cloud9-speaches-gpu-benchmark.lock
flock -n 8 || { echo "Speaches benchmark lock held; refusing" >&2; exit 4; }
exec 9>/run/cloud9-gpu.lock
flock -n 9 || { echo "Cloud9 global GPU lock held; refusing" >&2; exit 4; }

# Image user ubuntu:1000 is not a member of the render-device group.
# Passing --device alone does not grant rw access to /dev/dri/renderD128.
render_group="$(pct exec 410 -- stat -c %g /dev/dri/renderD128)"
[[ "$render_group" =~ ^[0-9]+$ ]] || { echo "Invalid CT410 render device group" >&2; exit 4; }

# Only after preflight and both GPU locks stage the benchmark.
tmp="/tmp/cloud9-vulkan-ab-${model}-$$"
cleanup() {
  pct exec 410 -- rm -f "$tmp/benchmark.py" 2>/dev/null || true
  pct exec 410 -- rmdir "$tmp" 2>/dev/null || true
}
trap cleanup EXIT
pct exec 410 -- mkdir -p "$tmp"
pct push 410 "$ROOT/scripts/cloud9_asr_vulkan_ab.py" "$tmp/benchmark.py"

# The benchmark inherits the existing GPU device; never hot-unbind or reset it.
# Production speaches containers are not restarted, modified or preempted.
pct exec 410 -- docker run --rm --name "cloud9-asr-vulkan-${model//[^a-z0-9-]/-}" \
  --network none --cpus 2 --memory 5g --memory-swap 5g \
  --pids-limit 128 --read-only --tmpfs /tmp:rw,nosuid,size=128m \
  --device /dev/dri/renderD128 --group-add "$render_group" \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env HF_HUB_OFFLINE=1 \
  --env OMP_NUM_THREADS=2 \
  --env OPENBLAS_NUM_THREADS=1 \
  --volume /srv/lxc/ia-compute/appdata/speaches/hf-hub:/home/ubuntu/.cache/huggingface/hub:ro \
  --volume /srv/lxc/ia-compute/data/bench-fixtures:/bench:ro \
  --volume "$tmp:/testlab:ro" \
  --entrypoint python ghcr.io/godsquantum/speaches-reloaded:latest-vulkan \
  -B /testlab/benchmark.py --backend vulkan --model "$model" --repeats 3

# The EXIT trap cleans staged files on success and failure.
