#!/usr/bin/env bash
cd "${0%/*}/../.."
echo 'Running tests'

python3 tests/test_commit.py
if [ $? -ne 0 ]; then
	echo 'Aborting commit (Attempting to commit a repository where the test suite fails)'
	echo 'Bypass with git commit --no-verify'
	exit 1
fi

exit 0
