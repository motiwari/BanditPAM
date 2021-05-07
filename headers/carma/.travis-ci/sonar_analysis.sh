#!/usr/bin/env bash
set -eo pipefail
[[ "$DEBUG_CI" == true ]] && set -x

case $TRAVIS_OS_NAME in
  linux)
    SONAR_WRAPPER=build-wrapper-linux-x86-64
    . .sonar_vars.sh
    ;;
  osx|windows)
    echo "$TRAVIS_OS_NAME not supported for SONAR analysis"
    exit 1
    ;;
  *)
    echo "Unknown OS [$TRAVIS_OS_NAME]"
    exit 1
    ;;
esac

${TRAVIS_BUILD_DIR}/.travis-ci/configure.sh
  
${SONAR_WRAPPER} --out-dir bw-output cmake --build build/ --config Release
sonar-scanner \
  -Dsonar.organization=rurlus \
  -Dsonar.projectKey=RUrlus_carma \
  -Dsonar.sources=include/.,tests/src/. \
  -Dsonar.host.url=https://sonarcloud.io \
  -Dsonar.cfamily.build-wrapper-output=bw-output \
  -Dsonar.login=890640f8296b91c98ccfa8ec2545220d68a4edb6
