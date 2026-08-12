#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
build_jobs="${ORB_SLAM2_BUILD_JOBS:-2}"

echo "Configuring and building Thirdparty/DBoW2 ..."
cmake -S "${project_root}/Thirdparty/DBoW2" \
      -B "${project_root}/Thirdparty/DBoW2/build" \
      -DCMAKE_BUILD_TYPE=Release
cmake --build "${project_root}/Thirdparty/DBoW2/build" --parallel "${build_jobs}"

echo "Configuring and building Thirdparty/g2o ..."
cmake -S "${project_root}/Thirdparty/g2o" \
      -B "${project_root}/Thirdparty/g2o/build" \
      -DCMAKE_BUILD_TYPE=Release
cmake --build "${project_root}/Thirdparty/g2o/build" --parallel "${build_jobs}"

echo "Uncompressing vocabulary if needed ..."
if [[ ! -f "${project_root}/Vocabulary/ORBvoc.txt" ]]; then
  tar -xf "${project_root}/Vocabulary/ORBvoc.txt.tar.gz" \
      -C "${project_root}/Vocabulary"
fi

echo "Configuring and building headless ORB_SLAM2 ..."
cmake -S "${project_root}" -B "${project_root}/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DORB_SLAM2_BUILD_VIEWER=OFF \
      -DBUILD_TESTING=ON
cmake --build "${project_root}/build" --parallel "${build_jobs}"
ctest --test-dir "${project_root}/build" --output-on-failure
