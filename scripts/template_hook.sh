#!/usr/bin/env bash
cd "${0%/*}/../.."
echo 'Running tests'

python3 tests/test_<stage>.py
if [ $? -ne 0 ]; then
	echo 'Aborting <stage> (Attempting to <stage> a repository where the test suite fails)'
	echo 'Bypass with git <stage> --no-verify'
	exit 1
fi

exit 0
