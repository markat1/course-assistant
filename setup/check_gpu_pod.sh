#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly repo_root

publish_check_script() {
  sudo k3s kubectl create configmap gpu-check \
    --from-file="$repo_root/setup/check_gpu_container.py" \
    --dry-run=client -o yaml | sudo k3s kubectl apply -f -
}

show_failure() {
  sudo k3s kubectl describe pod gpu-check
  sudo k3s kubectl logs pod/gpu-check || true
  exit 1
}

run_check_pod() {
  sudo k3s kubectl delete pod gpu-check --ignore-not-found
  sudo k3s kubectl apply -f "$repo_root/cluster/tests/gpu-check-pod.yaml"
  sudo k3s kubectl wait --for=jsonpath='{.status.phase}'=Succeeded \
    pod/gpu-check --timeout=180s || show_failure
}

main() {
  publish_check_script
  run_check_pod
  sudo k3s kubectl logs pod/gpu-check
  sudo k3s kubectl delete pod gpu-check
}

main "$@"