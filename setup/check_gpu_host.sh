#!/usr/bin/env bash
set -euo pipefail

echo "GPU and driver"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "Docker server"
docker version --format '{{.Server.Version}}'

echo "Docker Compose"
docker compose version