#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
revision=00c3bb11dc8b72fef0b26f2beb7a3e7f8c3a43ab
tree="$root/TensorFold"
patch="$root/patches/tensorfold-sm120.patch"

if [[ ! -e "$tree" ]]; then
    git init -q "$tree"
    git -C "$tree" remote add origin https://github.com/drowzeys/TensorFold.git
    git -C "$tree" fetch --depth 1 origin "$revision"
    git -C "$tree" checkout -q --detach FETCH_HEAD
fi

if [[ "$(git -C "$tree" rev-parse --verify HEAD)" != "$revision" ]]; then
    echo "Expected TensorFold revision $revision" >&2
    exit 1
fi
if git -C "$tree" apply --reverse --check "$patch" 2>/dev/null; then
    echo "TensorFold is already patched."
else
    git -C "$tree" apply --check "$patch"
    git -C "$tree" apply "$patch"
fi
