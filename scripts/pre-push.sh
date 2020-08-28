#!/usr/bin/env bash

echo "Running pre-push hook"
./scripts/pre-push-run-tests.sh

if [ $? -ne 0 ]; then
 echo "Tests must pass before push!"
 exit 1
fi
