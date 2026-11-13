#!/usr/bin/env bash
set -euo pipefail

root=${TF_SM120_ASSETS:-/port}
work=${TF_SM120_BUILD_ROOT:-$root}
tree=$work/TensorFold
out=$root/aot-sm120
ref=$root/ref
tool=$root/build/bin/tf-glm53-generate
context=32768
parallel=4
dcp=1
prompt_rows=8192
templates=(--template "$ref")
base_pack=
draft_args=(--dspark "$root/dspark-ft2")
case ${1:-dcp1} in
    dcp1) ;;
    dcp4-packed-a2a)
        previous=$root/aot-sm120-dcp4-prefill-8192
        out=$root/aot-sm120-dcp4-packed-a2a
        mkdir -p "$out"
        python3 -B - "$previous/check-decode.json" "$out/needs.json" <<'PY'
import json
import sys
from pathlib import Path

source = json.loads(Path(sys.argv[1]).read_text())
needs = [dict(n, fn="_merge_lse_packet", have=False)
         for n in source["needs"] if n["fn"] == "_merge_lse" and n["have"]]
assert needs
Path(sys.argv[2]).write_text(json.dumps({"needs": needs}) + "\n")
PY
        export PYTHONPATH="$root:$root/python-reference/src" TRITON_CACHE_DIR="$out/triton"
        python3 -B "$tree/tools/glm53/sweep_aot.py" --needs "$out/needs.json" \
            --out "$out/sweep" --merged "$out/packets"
        python3 -B "$tree/tools/glm53/sweep_aot.py" --merge-only --set "$previous/pack" \
            --set "$out/packets" --merged "$out/pack"
        exit 0
        ;;
    dcp4-256k)
        out=$root/aot-sm120-dcp4-256k
        context=262144
        parallel=1
        dcp=4
        prompt_rows=1024
        wide=$root/ref-dcp4
        mkdir -p "$wide"
        python3 -B "$root/catalog-template.py" "$root/aot-wide/aot.json" "$wide/manifest.json"
        templates=(--template "$wide" --template "$ref")
        ;;
    dcp4-fast-prefill)
        out=$root/aot-sm120-dcp4-prefill-8192
        context=262144
        parallel=1
        prompt_rows=8192
        base_pack=$root/aot-sm120-dcp4-256k/pack
        templates=(--template "$ref" --template "$root/ref-dcp4")
        ;;
    dcp4-fp8)
        out=$root/aot-sm120-dcp4-fp8
        tool=$root/build-dcp4-fp8-lmcache/bin/tf-glm53-generate
        context=524289
        parallel=1
        dcp=4
        prompt_rows=8192
        base_pack=$root/aot-sm120-dcp4-prefill-8192/pack
        templates=(--template "$ref" --template "$root/ref-dcp4")
        export TF_GLM53_FP8_KV=1
        ;;
    dcp4-fp8-4streams)
        out=$work/aot-sm120-dcp4-fp8-4streams
        tool=$work/build-dcp4-fp8-4streams/bin/tf-glm53-generate
        context=728889
        parallel=4
        dcp=4
        base_pack=$root/aot-sm120-dcp4-fp8/pack
        templates=(--template "$ref" --template "$root/ref-dcp4")
        draft_args=()
        export TF_GLM53_FP8_KV=1
        ;;
    dcp4-fp8-shared|dcp4-fp8-shared-dspark)
        tool=$work/build-dcp4-fp8-shared/bin/tf-glm53-generate
        context=728889
        parallel=4
        # DCP4 prefill also writes a rank-local BF16 cache. Enumerate DCP1 and DCP4 kernels.
        dcp=0
        if [[ $1 == dcp4-fp8-shared-dspark ]]; then
            out=$work/aot-sm120-dcp4-fp8-shared-dspark
            context=685001
        else
            out=$work/aot-sm120-dcp4-fp8-4streams
            draft_args=()
        fi
        templates=(--template "$ref" --template "$root/ref-dcp4")
        export TF_GLM53_FP8_KV=1 TF_GLM53_SHARED_KV=1
        ;;
    *) echo "Unknown kernel preset: $1" >&2; exit 2 ;;
esac
mkdir -p "$out"
test -s "$ref/manifest.json"
for rank in 0 1 2 3; do test -s "$ref/tiles-r$rank.json"; done
tiles="$ref/tiles-r0.json,$ref/tiles-r1.json,$ref/tiles-r2.json,$ref/tiles-r3.json"
shape=(--mode list-variants --model /model --world 4 --context "$context"
    --parallel "$parallel" --dcp "$dcp" --prompt-rows "$prompt_rows" --window 128 --k 0
    "${draft_args[@]}" --tiles "$tiles")
if [[ -n "$base_pack" ]]; then
    status=0
    "$tool" "${shape[@]}" --aot "$base_pack" --out "$out/needs.json" || status=$?
    [[ "$status" == 0 || "$status" == 3 ]]
    merged=$out/prefill
else
    "$tool" "${shape[@]}" --out "$out/needs.json"
    merged=$out/pack
fi

export PYTHONPATH="$work:$root:$root/python-reference/src"
export TRITON_CACHE_DIR="$out/triton"
python3 -B -c 'import torch; assert torch.cuda.get_device_capability() == (12, 0)'
python3 -B "$tree/tools/glm53/sweep_aot.py" --tools "$tree" \
    --needs "$out/needs.json" "${templates[@]}" --out "$out/sweep" --merged "$merged"
if [[ -n "$base_pack" ]]; then
    python3 -B "$tree/tools/glm53/sweep_aot.py" --merge-only --set "$base_pack" --set "$merged" \
        --merged "$out/pack"
    "$tool" --mode list-variants --model /model --world 4 --context "$context" \
        --parallel 1 --dcp 4 --prompt-rows 1024 --window 128 --k 0 \
        "${draft_args[@]}" --tiles "$tiles" --aot "$out/pack" --out "$out/check-decode.json"
fi
"$tool" "${shape[@]}" --aot "$out/pack" --out "$out/check.json"
