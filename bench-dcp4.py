"""Compare DCP4 with the recorded DCP1 workload on this host."""

import argparse
import json
import re
import time
import urllib.request

from launch_sm120 import PORT, PRESETS, ROOT, ready


DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))
DONE = re.compile(
    r"done req-(\d+) prompt=(\d+) cached=(\d+) tokens=(\d+) finish=\w+ "
    r"tok/s=([\d.]+) ttft=([\d.]+)s prefill=([\d.]+)s decode=([\d.]+)s.*?graphs=(\d+) eager=(\d+)"
)
PROMPTS = {
    "prose": "Write a 400-word essay on why the ocean is salty.",
    "code": "Write a Python class implementing an LRU cache with get and put, with type hints and docstrings, then a few unit tests.",
}


def request(preset, prompt, budget):
    log = preset.rank_logs / "rank-0.log"
    start_offset = log.stat().st_size
    body = {"model": "model", "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "reasoning_effort": "high", "max_tokens": budget}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    start = time.monotonic()
    with DIRECT.open(req, timeout=240) as response:
        data = json.load(response)
    wall = time.monotonic() - start
    with log.open("rb") as stream:
        stream.seek(start_offset)
        matches = DONE.findall(stream.read().decode(errors="replace"))
    usage = data["usage"]
    matches = [m for m in matches if (int(m[1]), int(m[3])) ==
               (usage["prompt_tokens"], usage["completion_tokens"])]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one completion log entry, saw {len(matches)}")
    _, prompt_tokens, cached, tokens, rate, ttft, prefill, decode, graphs, eager = matches[0]
    if not int(graphs) or int(eager):
        raise RuntimeError(f"Expected CUDA Graph decode, observed graphs={graphs} eager={eager}")
    return {"prompt_tokens": int(prompt_tokens), "cached_tokens": int(cached), "tokens": int(tokens),
            "tok_s": float(rate), "ttft_s": float(ttft), "prefill_s": float(prefill),
            "decode_s": float(decode), "wall_s": wall, "graphs": int(graphs), "eager": int(eager),
            "content": data["choices"][0]["message"].get("content")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=PRESETS, default="dcp4-fast-prefill")
    args = parser.parse_args()
    preset = PRESETS[args.preset]
    ready(preset)
    control = json.loads((ROOT / "benchmark.json").read_text())
    output = {"decode": {}}
    for name, prompt in PROMPTS.items():
        samples = [request(preset, prompt, 512) for _ in range(2)]
        if any(sample["tokens"] != 512 or not sample["content"] or "\ufffd" in sample["content"] for sample in samples):
            raise RuntimeError(f"Invalid completion for {name}")
        best = max(sample["tok_s"] for sample in samples)
        old = control["decode"][name]["sm120_tok_s"]
        output["decode"][name] = {"dcp1_tok_s": old, "dcp4_tok_s": best, "ratio": best / old, "samples": samples}
        print(f"{name}: {best:.1f} tok/s, {best / old:.1%} of DCP1", flush=True)
    document = " ".join(f"Entry {i}: the warehouse in sector {i % 37} holds {i * 7 % 911} crates."
                        for i in range(1400))
    for name, entry in (("cold", 1203), ("changed_tail", 77)):
        prompt = f"{document}\n\nHow many crates does entry {entry} hold? Answer with the number only."
        result = request(preset, prompt, 200)
        old = control["prefill"][name]
        if result["prompt_tokens"] != old["prompt_tokens"]:
            raise RuntimeError("Prompt token count changed from the DCP1 benchmark")
        result["correct"] = (result["content"] or "").strip().strip(".") == str(entry * 7 % 911)
        result["ratio"] = old["wall_s"] / result["wall_s"]
        result["prefill_ratio"] = old["prefill_s"] / result["prefill_s"]
        result["dcp1_wall_s"] = old["wall_s"]
        output[name] = result
        print(f"{name}: {result['wall_s']:.3f} s, {result['ratio']:.1%} of DCP1, "
              f"cached {result['cached_tokens']}, correct {result['correct']}", flush=True)
    marker = "MARBLE-LANTERN-7421"
    text = " ".join(f"Entry {i}: the warehouse in sector {i % 37} holds {i * 7 % 911} crates."
                    + (f" The unique verification phrase for this entire document is {marker}." if i == 1138 else "")
                    for i in range(2350))
    long = request(preset, text + " What is the unique verification phrase? Reply with the phrase only.", 384)
    long["correct"] = long["prompt_tokens"] > 32768 and marker in (long["content"] or "")
    output["long"] = long
    output["correct"] = output["cold"]["correct"] and output["changed_tail"]["correct"] and long["correct"]
    output["passed_70_percent"] = (
        output["correct"] and output["cold"]["cached_tokens"] == 0 and output["changed_tail"]["cached_tokens"] > 0
        and min(output["cold"]["ratio"], *(d["ratio"] for d in output["decode"].values())) >= 0.7
    )
    output_name = "benchmark-dcp4-fast.json" if args.preset == "dcp4-fast-prefill" else f"benchmark-{args.preset}.json"
    (ROOT / output_name).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    return 0 if output["passed_70_percent"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
