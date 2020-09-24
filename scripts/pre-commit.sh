#!/usr/bin/env bash
cd "${0%/*}/.."
echo 'Running tests'

git stashi > /dev/null
python3 -m unittest discover -p '*_commit.py'
if [ $? -ne 0 ]; then
	echo 'Aborting commit (Attempting to commit a repository where the test suite fails)'
	echo 'Bypass with git commit --no-verify'
	git stash pop > /dev/null
	exit 1
fi
git stash pop > /dev/null

exit 0
