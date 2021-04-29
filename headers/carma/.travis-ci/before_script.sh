#!/usr/bin/env bash
set -e
[[ "$DEBUG_CI" == true ]] && set -x

echo "PY_CMD = ${PY_CMD} [$(command -v ${PY_CMD})] [$(${PY_CMD} --version)]"
echo "CXX = ${CXX} [$(command -v ${CXX})]"
echo "CC = ${CC} [$(command -v ${CC})]"
echo "CXXFLAGS = ${CXXFLAGS}"
echo "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}"
