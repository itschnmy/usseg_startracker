#!/usr/bin/env bash
# Host interface: self-test then process camera input; outputs survive container exit.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: bash scripts/run_test.sh INPUT_DIR OUTPUT_DIR CONFIG_DIR" >&2
  exit 2
fi
command -v podman >/dev/null || { echo "Podman is required." >&2; exit 2; }
input_dir=$(realpath -e -- "$1")
mkdir -p -- "$2"
output_dir=$(realpath -e -- "$2")
config_dir=$(realpath -e -- "$3")
image_name=${USSEG_IMAGE:-localhost/usseg:prod-version2}
# --userns=keep-id runs with the host user's UID/GID; output must be host-writable.
common=(--rm --pull=never --network=none --read-only --userns=keep-id
  --cap-drop=all --security-opt=no-new-privileges --stop-timeout=10
  --tmpfs /tmp:rw,nosuid,nodev,size=256m
  --mount "type=bind,src=$input_dir,dst=/input,ro"
  --mount "type=bind,src=$output_dir,dst=/output"
  --mount "type=bind,src=$config_dir,dst=/config,ro")
# Self-test has known profiles and an image-internal database; external camera
# profiles are exercised by the subsequent batch, not by substituting lab truth.
podman run "${common[@]}" "$image_name" self-test --config /app/configs/runtime/development.json
exec podman run "${common[@]}" "$image_name" batch --config /config/runtime.json
