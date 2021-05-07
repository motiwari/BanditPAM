#!/usr/bin/env bash
set -eo pipefail
[[ "$DEBUG_CI" == true ]] && set -x

CMAKE_EXTRA_ARGS+=" -DBUILD_TESTS=ON"

if [ -n "$CPP" ]; then CPPSTD=-std=c++$CPP; fi

mkdir build
cd build
cmake ${CMAKE_EXTRA_ARGS} \
  -DPYTHON_PREFIX_PATH=${PYTHON_PREFIX_PATH} \
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
  -DPYBIND11_PYTHON_VERSION=$PYTHON_VERSION \
  -DPYBIND11_CPP_STANDARD=$CPPSTD \
  ..
