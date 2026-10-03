#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 3 ]] || { echo "usage: $0 AUDIO.wav REFERENCE.txt OUTPUT.json" >&2; exit 2; }
AUDIO=$1
REFERENCE=$2
OUTPUT=$3
BENCH_PY=${CLOUD9_STT_BENCH_PY:-/srv/lxc/ia-compute/data/bench-tools/cloud9_fr_stt_benchmark.py}
LOCK=/run/cloud9-gpu.lock
exec 9>"$LOCK"
flock -n 9 || { echo "Cloud9 GPU lock busy" >&2; exit 4; }

ROUTER_SOCKET=cloud9-engine-router-proxy.socket
ROUTER_PROXY=cloud9-engine-router-proxy.service
ROUTER=cloud9-engine-router.service
EMBED=cloud9-embedding.service
LEMOND=lemond.service
FRONT_SOCKETS=(comfyui-proxy.socket autopublisher-image-proxy.socket)
FRONT_PROXIES=(comfyui-proxy.service autopublisher-image-proxy.service)
IMAGE_SERVICES=(comfyui.service autopublisher-image.service)

was_router_socket=0; was_embed=0; was_lemond=0; was_voice=0
declare -a restore_sockets=()
systemctl is-active --quiet "$ROUTER_SOCKET" && was_router_socket=1 || true
systemctl is-active --quiet "$EMBED" && was_embed=1 || true
systemctl is-active --quiet "$LEMOND" && was_lemond=1 || true
docker inspect -f '{{.State.Running}}' voicestudio 2>/dev/null | grep -qx true && was_voice=1 || true
for u in "${FRONT_SOCKETS[@]}"; do systemctl is-active --quiet "$u" && restore_sockets+=("$u") || true; done

restore() {
  rc=$?
  trap - EXIT INT TERM
  (( was_voice )) && docker start voicestudio >/dev/null 2>&1 || true
  (( was_embed )) && systemctl start "$EMBED" >/dev/null 2>&1 || true
  (( was_lemond )) && systemctl start "$LEMOND" >/dev/null 2>&1 || true
  (( was_router_socket )) && systemctl start "$ROUTER_SOCKET" >/dev/null 2>&1 || true
  for u in "${restore_sockets[@]}"; do systemctl start "$u" >/dev/null 2>&1 || true; done
  exit "$rc"
}
trap restore EXIT INT TERM

for u in "${IMAGE_SERVICES[@]}"; do
  systemctl is-active --quiet "$u" && { echo "$u is active; refusing benchmark" >&2; exit 4; }
done
ss -Htn state established | awk '$4 ~ /:8005$/ || $5 ~ /:8005$/ {x=1} END{exit x?0:1}' &&
  { echo "Speaches has an established client; refusing benchmark" >&2; exit 4; } || true
if (( was_voice )); then
  ss -Htn state established | awk '$4 ~ /:(3900|7443)$/ || $5 ~ /:(3900|7443)$/ {x=1} END{exit x?0:1}' &&
    { echo "VoiceStudio has an established client; refusing benchmark" >&2; exit 4; } || true
fi

systemctl stop "${FRONT_SOCKETS[@]}" "${FRONT_PROXIES[@]}" >/dev/null 2>&1 || true
systemctl stop "$ROUTER_SOCKET" "$ROUTER_PROXY" "$ROUTER" >/dev/null 2>&1 || true
systemctl stop "$EMBED" "$LEMOND" >/dev/null 2>&1 || true
(( was_voice )) && docker stop --time 20 voicestudio >/dev/null 2>&1 || true

docker inspect -f '{{.State.Health.Status}}' speaches 2>/dev/null | grep -qx healthy ||
  { echo "Speaches is not healthy" >&2; exit 5; }

python3 "$BENCH_PY" --audio "$AUDIO" --reference "$REFERENCE" --output "$OUTPUT" --passes 2
