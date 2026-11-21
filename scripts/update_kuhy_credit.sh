#!/bin/bash
# Vendors the kuhy credit web component from utils@kuhy-credit-v0.3.0.
# Re-run after bumping TAG; never edit the vendored files by hand.
set -euo pipefail
cd "$(dirname "$0")/.."
readonly TAG="kuhy-credit-v0.3.0"
readonly REPO="https://github.com/kuhyx/utils"
readonly SRC="kuhy_credit/web/src"
readonly DEST="src/mangatl/web/vendor/kuhy-credit"
cache="$(mktemp -d)"
trap 'rm -rf "$cache"' EXIT
git clone -q --depth 1 --branch "$TAG" "$REPO" "$cache"
mkdir -p "$DEST"
cp "$cache/$SRC/kuhy-credit.js" "$cache/$SRC/credit-data.js" "$DEST/"
