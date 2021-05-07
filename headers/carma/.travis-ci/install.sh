#!/usr/bin/env bash
set -eo pipefail
[[ "$DEBUG_CI" == true ]] && set -x

case $TRAVIS_OS_NAME in
  linux|osx)    
    ;;
  windows)
    ;;
  *)
    echo "Unknown OS [$TRAVIS_OS_NAME]"
    exit 1
    ;;
esac

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
${PY_CMD} get-pip.py # or use apt-get install python3-pip
${PY_CMD} -m pip install --progress-bar off pip --upgrade
${PY_CMD} -m pip install --progress-bar off pytest numpy cmake --upgrade

if [ "$VALGRIND" = true ]; then 
  ${PY_CMD} -m pip install pytest-valgrind
fi
