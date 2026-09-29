#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly repo_root

label_gpu_node() {
  sudo k3s kubectl label node "$(hostname)" gpu=on --overwrite
}

install_hami() {
  sudo k3s kubectl apply -f "$repo_root/cluster/hami/hami.yaml"
}

wait_for_component() {
  local resource="$1"

  sudo k3s kubectl -n kube-system wait --for=create "$resource" --timeout=300s
  sudo k3s kubectl -n kube-system rollout status "$resource" --timeout=300s
}

main() {
  label_gpu_node
  install_hami
  wait_for_component deployment/hami-scheduler
  wait_for_component daemonset/hami-device-plugin
}

main "$@"
