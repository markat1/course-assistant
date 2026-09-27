#!/usr/bin/env bash
set -euo pipefail

target="${1:?Usage: bash setup/ssh_gpu.sh user@host /absolute/path/to/key}"
key="${2:?Provide the absolute path to your SSH key}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
revision="$(git -C "$root" rev-parse --verify HEAD)"

exec ssh -t -i "$key" \
  -o IdentitiesOnly=yes \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:18780:127.0.0.1:8780 \
  -L 127.0.0.1:23000:127.0.0.1:13000 \
  -L 127.0.0.1:29090:127.0.0.1:19090 \
  -L 127.0.0.1:28001:127.0.0.1:8001 \
  -L 127.0.0.1:28002:127.0.0.1:8002 \
  -- "$target" \
  "cd ~/course-assistant/releases/$revision && export COMPOSE_PROJECT_NAME=course-assistant && exec bash -l"