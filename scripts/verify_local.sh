#!/usr/bin/env bash
# Ground-side suite. Runtime container acceptance is scripts/run_test.sh instead.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
if [[ $# -ne 1 || -e "$1" ]]; then
  echo "Usage: bash scripts/verify_local.sh NEW_OUTPUT_DIR" >&2
  exit 2
fi
output_dir=$1
mkdir -p -- "$output_dir"
test_python=${USSEG_PYTHON:-.venv/bin/python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$test_python" -m pytest tests/python -q --junitxml="$output_dir/pytest-junit.xml" 2>&1 | tee "$output_dir/pytest.log"
"$test_python" -m usseg_runtime.cli self-test --config configs/runtime/development.json \
  --output "$output_dir" 2>&1 | tee "$output_dir/self-test.log"
"$test_python" -m examples.run_batch --output "$output_dir/camera" 2>&1 | tee "$output_dir/camera.log"
bash scripts/verify_cpp.sh "$output_dir/cpp"
echo "Ground validation complete: $output_dir"
