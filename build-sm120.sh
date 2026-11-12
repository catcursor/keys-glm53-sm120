#!/usr/bin/env bash
set -euo pipefail

root=${1:?usage: build-sm120.sh /port}
cd "$root/TensorFold"
export PATH="$root/zig:$PATH"
export ZIG_LOCAL_CACHE_DIR="$root/zig-cache/local"
export ZIG_GLOBAL_CACHE_DIR="$root/zig-cache/global"
nvcc=/usr/local/cuda/bin/nvcc
prefix=${TF_SM120_PREFIX:-"$root/build"}
jobs=${JOBS:-8}

test "$(uname -m)" = x86_64
test "$(zig version)" = 0.17.0
"$nvcc" --list-gpu-code | grep -Fx sm_120
zig build -Dnvcc="$nvcc" -Dsm=120 -Doptimize=safe --prefix "$prefix" -j"$jobs" fatbins
zig build -Dfatbins="$prefix/fatbin" -Dsm=120 -Doptimize=safe --prefix "$prefix" -j"$jobs" \
    native glm53-lib tf-glm53-generate
zig build -Dnvcc="$nvcc" -Dsm=120 -Doptimize=safe --prefix "$prefix" -j"$jobs" \
    test-glm53 test-cuda-host
"$prefix/native/bin/tensorfold-native" --version
