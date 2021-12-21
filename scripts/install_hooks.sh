#!/bin/bash

[ ! -e ./.git/hooks/pre-commit ] || rm ./.git/hooks/pre-commit
[ ! -e ./.git/hooks/pre-push ] || rm ./.git/hooks/pre-push

GIT_DIR=$(git rev-parse --git-dir)

echo "Installing hooks..."
cp ./scripts/template_hook.sh $GIT_DIR/hooks/pre-commit
chmod +x $GIT_DIR/hooks/pre-commit

sed -i -e 's/<stage>/commit/g' $GIT_DIR/hooks/pre-commit

cp ./scripts/template_hook.sh $GIT_DIR/hooks/pre-push
chmod +x $GIT_DIR/hooks/pre-push

sed -i -e 's/<stage>/push/g' $GIT_DIR/hooks/pre-push

echo "Done!"
