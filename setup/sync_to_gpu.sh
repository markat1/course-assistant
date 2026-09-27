#!/usr/bin/env bash
set -euo pipefail

target="${1:?Usage: bash setup/sync_to_gpu.sh user@host /absolute/path/to/key}"
key="${2:?Provide the absolute path to your SSH key}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if ! git diff --quiet HEAD --; then
  echo "Commit tracked changes before syncing." >&2
  exit 1
fi

revision="$(git rev-parse --verify HEAD)"
remote_dir="course-assistant/releases/$revision"

echo "Uploading commit $revision to $target:~/$remote_dir"
git archive --format=tar "$revision" |
  ssh -i "$key" -o IdentitiesOnly=yes -- "$target" \
    "mkdir -p ~/$remote_dir && tar -xf - -C ~/$remote_dir"

echo "Uploaded committed files. Untracked files and local .env were not included."