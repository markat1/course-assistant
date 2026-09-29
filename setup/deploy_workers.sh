#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly repo_root

main() {
  sudo k3s kubectl apply -f "$repo_root/cluster/workers/sglang.yaml"
  sudo k3s kubectl rollout status statefulset/sglang --timeout=1800s
  sudo k3s kubectl get pods -l app=sglang -o wide
}

main "$@"