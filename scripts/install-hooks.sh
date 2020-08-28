#!/usr/bin/env bash

[ ! -e ./.git/hooks/pre-commit ] || rm ./.git/hooks/pre-commit
[ ! -e ./.git/hooks/pre-push ] || rm ./.git/hooks/pre-push

GIT_DIR=$(git rev-parse --git-dir)

echo "Installing hooks..."
ln -s ../../scripts/pre-commit.sh $GIT_DIR/hooks/pre-commit
ln -s ../../scripts/pre-push.sh $GIT_DIR/hooks/pre-push
echo "Done!"
