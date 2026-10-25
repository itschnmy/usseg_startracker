#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
asset_dir="$repo_root/data"
asset="$asset_dir/default_database.npz"
expected_sha256="4fad84d8d7d922e240abbb557dd6329ed94468c76657eebd4a3d9d042ab0f412"

if [[ ! -f "$asset" ]]; then
    mkdir -p "$asset_dir"
    temp_asset=$(mktemp "$asset_dir/.database-download-XXXXXX")
    trap 'rm -f -- "$temp_asset"' EXIT
    git show origin/main:identificator/default_database.npz > "$temp_asset"
    temp_sha256=$(sha256sum "$temp_asset" | awk '{print $1}')
    if [[ "$temp_sha256" != "$expected_sha256" ]]; then
        echo "Downloaded database checksum mismatch: $temp_sha256" >&2
        exit 1
    fi
    mv -- "$temp_asset" "$asset"
fi

actual_sha256=$(sha256sum "$asset" | awk '{print $1}')
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    echo "Database checksum mismatch: $actual_sha256" >&2
    exit 1
fi

echo "Database ready: $asset"
