#!/usr/bin/env bash
set -euo pipefail

readonly K3S_VERSION="v1.35.8+k3s1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly repo_root



fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

check_prerequisites() {
  command -v nvidia-container-runtime >/dev/null ||
    fail "NVIDIA Container Runtime is missing."

  if command -v k3s >/dev/null; then
    local installed_version
    installed_version="$(k3s --version | awk 'NR == 1 {print $3}')"

    [[ "$installed_version" == "$K3S_VERSION" ]] ||
      fail "Expected k3s $K3S_VERSION; found $installed_version."
  fi
}

install_config() {
  local source="$repo_root/cluster/k3s/config.yaml"
  local destination="/etc/rancher/k3s/config.yaml"

  [[ -f "$source" ]] || fail "Missing configuration: $source"

  if sudo test -e "$destination"; then
    sudo cmp -s "$source" "$destination" ||
      fail "Existing k3s configuration differs. Review it first."
  fi

  sudo install -d -m 0755 /etc/rancher/k3s
  sudo install -m 0600 "$source" "$destination"
}

install_k3s() {
  if command -v k3s >/dev/null; then
    printf 'k3s %s is already installed.\n' "$K3S_VERSION"
    return
  fi

  (
    installer_dir="$(mktemp -d)"
    trap 'rm -rf "$installer_dir"' EXIT

    curl -fL \
      "https://raw.githubusercontent.com/k3s-io/k3s/${K3S_VERSION}/install.sh" \
      -o "$installer_dir/install.sh"

    sudo env INSTALL_K3S_VERSION="$K3S_VERSION" \
      sh "$installer_dir/install.sh" server
  )
}

wait_for_node() {
  local deadline=$((SECONDS + 180))

  while (( SECONDS < deadline )); do
    if sudo k3s kubectl wait \
      --for=condition=Ready node --all \
      --timeout=5s --request-timeout=5s; then
      return
    fi

    sleep 2
  done

  fail "Kubernetes node did not become Ready within the startup window."
}

wait_for_runtimeclass() {
  local deadline=$((SECONDS + 60))

  while (( SECONDS < deadline )); do
    if sudo k3s kubectl get runtimeclass nvidia \
      --request-timeout=5s >/dev/null 2>&1; then
      return
    fi

    sleep 2
  done

  fail "RuntimeClass nvidia was not created within the startup window."
}

check_cluster() {
  sudo systemctl start k3s
  wait_for_node
  wait_for_runtimeclass
  sudo k3s kubectl get nodes -o wide
  sudo k3s kubectl get runtimeclass nvidia
}

main() {
  check_prerequisites
  install_config
  install_k3s
  check_cluster
}

main "$@"