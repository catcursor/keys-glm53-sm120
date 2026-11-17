#!/usr/bin/env bash
set -euo pipefail

port=${1:?missing HTTP port}
context=${2:-32768}
parallel=${3:-4}
native=${TF_GLM53_NATIVE:-/port/build/native/bin/tensorfold-native}
export TF_GLM53_NCCL_LIB
TF_GLM53_NCCL_LIB=$(find /usr/local /opt /usr/lib -name libnccl.so.2 -print -quit)
test -n "$TF_GLM53_NCCL_LIB"
export LD_LIBRARY_PATH="$(dirname "$TF_GLM53_NCCL_LIB"):${LD_LIBRARY_PATH:-}"

pids=()
stop_ranks() {
    if (( ${#pids[@]} )); then
        kill "${pids[@]}" 2>/dev/null || true
    fi
}
trap stop_ranks EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

for rank in 0 1 2 3; do
    affinity_args=()
    if [[ -n ${TF_GLM53_RANK_CPUSETS:-} ]]; then
        IFS=';' read -r -a rank_cpus <<< "$TF_GLM53_RANK_CPUSETS"
        if (( ${#rank_cpus[@]} != 4 )); then
            echo "TF_GLM53_RANK_CPUSETS must contain four semicolon-separated CPU sets" >&2
            exit 1
        fi
        affinity_args=(taskset -c "${rank_cpus[$rank]}")
    fi
    learn_args=()
    if [[ -n ${TF_GLM53_LMCACHE_PORT:-} ]]; then
        learn_dir=/rank-logs/lmcache-index/rank-$rank
        mkdir -p -m 700 "$learn_dir"
        learn_args=(--learn-dir "$learn_dir" --learn-gib "${TF_GLM53_LMCACHE_LEARN_GIB:-4}")
    fi
    TF_GLM53_RANK=$rank TF_GLM53_DEVICE=$rank TF_GLM53_TILES=/port/ref/tiles-r$rank.json \
        "${affinity_args[@]}" "$native" serve /model --host 127.0.0.1 --port "$port" --name model \
        --context "$context" --parallel "$parallel" --thinking --reasoning-effort high --no-update-check "${learn_args[@]}" \
        >"/rank-logs/rank-$rank.log" 2>&1 &
    pids+=("$!")
done

status=0
wait -n "${pids[@]}" || status=$?
if (( status == 0 )); then
    echo "A TensorFold rank exited while serving" >&2
    exit 1
fi
exit "$status"
