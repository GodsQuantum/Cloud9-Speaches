#!/usr/bin/env bash
set -euo pipefail
docker compose -f compose.vulkan.yaml run --rm --entrypoint python speaches scripts/preload-models.py "$@"
