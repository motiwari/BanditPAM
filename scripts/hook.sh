#!/usr/bin/env bash
cd "${0%/*}/../.."
echo 'Running tests'

python3 tests/test_push.py
if [ $? -ne 0 ]; then
	echo 'Aborting push (Attempting to push a repository where the test suite fails)'
	echo 'Bypass with git push --no-verify'
	exit 1
fi

exit 0
