#!/usr/bin/env bash 
set -eo pipefail
[[ "$DEBUG_CI" == true ]] && set -x


case $TRAVIS_OS_NAME in
  linux)    
    export SONAR_SCANNER_VERSION=4.2.0.1873
    export SONAR_SCANNER_HOME=$HOME/.sonar/sonar-scanner-$SONAR_SCANNER_VERSION-linux
    curl --create-dirs -sSLo $HOME/.sonar/sonar-scanner.zip https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-$SONAR_SCANNER_VERSION-linux.zip 
    unzip -o $HOME/.sonar/sonar-scanner.zip -d $HOME/.sonar/
    
    
    echo "export PATH=$SONAR_SCANNER_HOME/bin:\$PATH" > .sonar_vars.sh
    echo "export SONAR_SCANNER_OPTS=-server" >> .sonar_vars.sh
    
    curl --create-dirs -sSLo $HOME/.sonar/build-wrapper-linux-x86.zip https://sonarcloud.io/static/cpp/build-wrapper-linux-x86.zip
    unzip -o $HOME/.sonar/build-wrapper-linux-x86.zip -d $HOME/.sonar/
    
    echo "export PATH=\$HOME/.sonar/build-wrapper-linux-x86:\$PATH" >> .sonar_vars.sh
    ;;
  osx|windows)
    echo "$TRAVIS_OS_NAME not supported for SONAR installation"
    exit 1
    ;;
  *)
    echo "Unknown OS [$TRAVIS_OS_NAME]"
    exit 1
    ;;
esac

