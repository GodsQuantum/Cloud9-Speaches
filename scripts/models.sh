#!/usr/bin/env bash
set -euo pipefail

container="${SPEACHES_CONTAINER:-speaches}"
lean=(
  "handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf"
  "speaches-ai/Kokoro-82M-v1.0-ONNX-int8"
  "speaches-ai/piper-fr_FR-tom-medium"
)
recommended=(
  "${lean[@]}"
  "handy-computer/Qwen3-ASR-0.6B-gguf"
  "deepdml/faster-whisper-large-v3-turbo-ct2"
)
full=(
  "${recommended[@]}"
  "handy-computer/Qwen3-ASR-1.7B-gguf"
  "handy-computer/parakeet-tdt-0.6b-v3-gguf"
  "handy-computer/diar_streaming_sortformer_4spk-v2.1-gguf"
)

preset="${1:-}"
if [[ -z "$preset" ]]; then
  printf '%s\n' "Download a model pack:" "  1) lean" "  2) recommended" "  3) full"
  read -r -p "Choice [2]: " choice
  case "${choice:-2}" in
    1) preset=lean ;;
    2|"") preset=recommended ;;
    3) preset=full ;;
    *) echo "Invalid choice" >&2; exit 2 ;;
  esac
fi
case "$preset" in
  lean) models=("${lean[@]}") ;;
  recommended) models=("${recommended[@]}") ;;
  full) models=("${full[@]}") ;;
  custom)
    shift
    (( $# > 0 )) || { echo "Usage: $0 custom <hf-model-id> [...]" >&2; exit 2; }
    models=("$@")
    ;;
  *) echo "Unknown preset: $preset (lean|recommended|full|custom)" >&2; exit 2 ;;
esac

docker inspect "$container" >/dev/null 2>&1 || {
  echo "Container '$container' not found. Start Speaches Reloaded first." >&2
  exit 1
}
[[ "$(docker inspect -f '{{.State.Running}}' "$container")" == "true" ]] || {
  echo "Container '$container' is not running." >&2
  exit 1
}

printf 'Downloading %d model repositories into the persistent Hugging Face cache...\n' "${#models[@]}"
docker exec "$container" python /home/ubuntu/speaches/scripts/preload-models.py "${models[@]}"
echo "Model pack '$preset' is ready."
