#!/usr/bin/env bash
set -e
cd "${0%/*}/.."
echo "Running tests"
echo "............................"
cd "./build/tests"
./runTest
RESULT=$?
if [[ $RESULT -ne 0 ]]
then
  echo "Failed!"
  exit 1
else
  echo "Successful!"
  exit 0
fi
