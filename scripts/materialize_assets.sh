#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
asset="$repo_root/identificator/default_database.npz"
expected_sha256="4fad84d8d7d922e240abbb557dd6329ed94468c76657eebd4a3d9d042ab0f412"

if [[ ! -f "$asset" ]]; then
    git show origin/main:identificator/default_database.npz > "$asset"
fi

actual_sha256=$(sha256sum "$asset" | awk '{print $1}')
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    echo "Database checksum mismatch: $actual_sha256" >&2
    exit 1
fi

echo "Database ready: $asset"
