#!/usr/bin/env bash
set -Eeuo pipefail
nodes=${SPEACHES_CLUSTER_NODES:-${1:-}}
[[ -n "$nodes" ]] || { echo "Usage: SPEACHES_CLUSTER_NODES=http://node1:8000,http://node2:8000 $0" >&2; exit 64; }

IFS=',' read -r -a list <<<"$nodes"
fail=0
for base in "${list[@]}"; do
  base=${base%/}
  echo "== $base =="
  if curl -fsS --connect-timeout 2 --max-time 8 "$base/health" >/dev/null; then
    echo "health: OK"
  else
    echo "health: FAIL" >&2; fail=1; continue
  fi
  if curl -fsS --connect-timeout 2 --max-time 8 "$base/v1/models" >/tmp/speaches-models.$$; then
    python3 - /tmp/speaches-models.$$ <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    ids=[x.get("id","?") for x in d.get("data",[])]
    print("models:", ", ".join(ids[:12]) + (" ..." if len(ids)>12 else ""))
except Exception as e:
    print("models: unreadable:", e)
PY
  else
    echo "models: probe unavailable"
  fi
done
rm -f /tmp/speaches-models.$$
exit "$fail"
