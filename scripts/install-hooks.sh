#!/usr/bin/env bash

[ ! -e ./.git/hooks/pre-commit ] || rm ./.git/hooks/pre-commit
[ ! -e ./.git/hooks/pre-push ] || rm ./.git/hooks/pre-push

GIT_DIR=$(git rev-parse --git-dir)

echo "Installing hooks..."
cp ./scripts/hook.sh $GIT_DIR/hooks/pre-commit
chmod +x $GIT_DIR/hooks/pre-commit
sed -i '' 's/commit/push/g' scripts/hook.sh
cp ./scripts/hook.sh $GIT_DIR/hooks/pre-push
chmod +x $GIT_DIR/hooks/pre-push

echo "Done!"
