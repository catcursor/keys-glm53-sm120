"""Check actual DCP4 FP8 inference and an LMCache restore after a GPU cache flush."""
import json
import runpy

from launch_sm120 import PRESETS, ROOT, ready


def main():
    preset = PRESETS["dcp4-fp8-lmcache"]
    ready(preset)
    request = runpy.run_path(str(ROOT / "bench-dcp4.py"))["request"]
    marker = "NEON-CASCADE-9514"
    document = " ".join(
        f"Entry {i}: the warehouse in sector {i % 37} holds {i * 7 % 911} crates."
        + (f" The unique verification phrase for this entire document is {marker}." if i == 420 else "")
        for i in range(1400)
    )
    queries = (
        " What is the unique verification phrase in this document? Reply with the phrase only.",
        " Return the unique verification phrase from the document as your only answer.",
    )
    log = preset.rank_logs / "rank-0.log"
    before = log.stat().st_size
    cold = request(preset, document + queries[0], 256)
    with log.open("rb") as stream:
        stream.seek(before)
        saved = stream.read().decode(errors="replace")
    if cold["cached_tokens"] != 0 or marker not in (cold["content"] or "") or "LMCache: stored a prefix" not in saved:
        raise RuntimeError(f"Cold prompt failed to store a correct answer in LMCache: {cold}")
    flush = preset.rank_logs / ".flush-gpu-prefix"
    flush.touch()
    before = log.stat().st_size
    warm = request(preset, document + queries[1], 256)
    with log.open("rb") as stream:
        stream.seek(before)
        restored = stream.read().decode(errors="replace")
    if flush.exists() or warm["cached_tokens"] < 16384 or marker not in (warm["content"] or ""):
        raise RuntimeError(f"LMCache reuse did not produce the expected response: {warm}")
    if "LMCache restored" not in restored:
        raise RuntimeError("No evidence of restoring the GPU state from LMCache")
    result = {"context": preset.context, "cold": cold, "after_gpu_cache_flush": warm}
    (ROOT / "benchmark-dcp4-fp8-lmcache-restore.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
