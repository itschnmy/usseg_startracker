#!/usr/bin/env bash
# Separate ground-side pipeline; never confused with Python image solver accuracy.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
if [[ $# -ne 1 ]]; then echo "Usage: bash scripts/verify_cpp.sh OUTPUT_DIR" >&2; exit 2; fi
mkdir -p -- "$1"
output_dir=$(realpath -e -- "$1")
if ! command -v g++ >/dev/null; then
  echo "SKIP: C++ ground tests require g++ (not required by Python runtime)." | tee "$output_dir/status.log"
  exit 0
fi
sources=(models/cpp/estimator.cpp models/cpp/function.cpp models/cpp/CentroidMatcher.cpp
  models/cpp/StarCatalog.cpp models/cpp/PairCatalog.cpp models/cpp/VotingIdentifier.cpp)
for test_name in lis tracking; do
  g++ -std=c++14 -O2 -I models/cpp "tests/cpp/test_$test_name.cpp" "${sources[@]}" -o "$output_dir/test_$test_name"
  (cd data && "$output_dir/test_$test_name") 2>&1 | tee "$output_dir/$test_name.log"
done
if command -v pkg-config >/dev/null && pkg-config --exists opencv4; then
  # Compiler flags come from the installed pkg-config module, not user input.
  read -r -a cv_flags <<< "$(pkg-config --cflags opencv4)"
  read -r -a cv_libs <<< "$(pkg-config --libs opencv4)"
  g++ -std=c++14 -O2 -I models/cpp/ImagePreprocessor "${cv_flags[@]}" \
    tests/cpp/test_image_preprocessing.cpp models/cpp/ImagePreprocessor/ImagePreprocessor.cpp \
    "${cv_libs[@]}" -o "$output_dir/test_image_preprocessing"
  (cd "$output_dir" && ./test_image_preprocessing "$repo_dir/dataset/calibration/openmv/session_01/03_calibrated.bmp") \
    2>&1 | tee "$output_dir/image_preprocessing.log"
else
  echo "SKIP: OpenCV C++ headers/libs absent; Python OpenCV wheel does not provide this target." \
    | tee "$output_dir/image_preprocessing.log"
fi
echo "C++ LIS/tracking completed; read preprocessing log for optional target." | tee "$output_dir/status.log"
