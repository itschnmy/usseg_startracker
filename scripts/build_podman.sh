#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
command -v podman >/dev/null || { echo "Podman is required for build; install on the build host." >&2; exit 2; }
image_name=${USSEG_IMAGE:-localhost/usseg:prod-version2}
build_python=${USSEG_PYTHON:-python3.10}
source_revision=$(git rev-parse HEAD)
if [[ -n $(git status --porcelain) ]]; then
  echo "Commit reviewed changes before building a traceable image." >&2
  exit 2
fi
"$build_python" -c 'from pathlib import Path; from usseg_runtime.config import load_config; load_config(Path("configs/runtime/development.json"))'
podman build --format oci --build-arg "SOURCE_REVISION=$source_revision" -t "$image_name" -f Containerfile .
destination=${1:-out/podman-release}
mkdir -p -- "$destination"
podman image inspect "$image_name" > "$destination/image-inspect.json"
podman save --format oci-archive --output "$destination/usseg-prod-version2.oci.tar" "$image_name"
(cd "$destination" && sha256sum usseg-prod-version2.oci.tar image-inspect.json > SHA256SUMS)
echo "OCI archive and manifest: $destination"
