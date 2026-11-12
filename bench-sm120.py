"""Compare the native SM120 server against the published four-Spark workload."""

import json
import re
import time
import urllib.request

from launch_sm120 import PORT, RANK_LOGS, ROOT, ready


PROMPTS = {
    "prose": "Write a 400-word essay on why the ocean is salty.",
    "code": "Write a Python class implementing an LRU cache with get and put, with type hints and docstrings, then a few unit tests.",
}
SPARK_TOK_S = {"prose": 38.3, "code": 56.3}
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))
LOG_ROW = re.compile(
    r"done req-(\d+) prompt=(\d+) cached=(\d+) tokens=(\d+).*?tok/s=([\d.]+)"
    r" ttft=([\d.]+)s prefill=([\d.]+)s decode=([\d.]+)s.*?graphs=(\d+) eager=(\d+)"
)


def rows():
    log = (RANK_LOGS / "rank-0.log").read_text(errors="replace")
    return {int(row[0]): row for row in LOG_ROW.findall(log)}


def request(prompt, count, **extra):
    previous = rows()
    data = {"model": "model", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": count, "temperature": 0, "reasoning_effort": "high", **extra}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    begin = time.monotonic()
    with DIRECT.open(req, timeout=900) as response:
        result = json.load(response)
    wall = time.monotonic() - begin
    choice = result["choices"][0]
    deadline = time.monotonic() + 2
    while True:
        delta = [row for key, row in rows().items() if key not in previous]
        if delta or time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    if len(delta) != 1:
        raise RuntimeError(f"Expected one completed rank-0 request, saw {len(delta)}")
    _, prompt_tokens, cached, tokens, rate, ttft, prefill, decode, graphs, eager = delta[0]
    return {
        "prompt_tokens": int(prompt_tokens), "cached_tokens": int(cached), "tokens": int(tokens),
        "tok_s": float(rate), "ttft_s": float(ttft), "prefill_s": float(prefill), "decode_s": float(decode),
        "wall_s": round(wall, 3), "graphs": int(graphs), "eager": int(eager),
        "finish_reason": choice["finish_reason"], "message": choice["message"],
    }


def benchmark():
    ready()
    smoke = request("What is the capital of France? One word.", 256)
    if "paris" not in (smoke["message"].get("content") or "").lower():
        raise RuntimeError("The basic correctness check did not return Paris")
    results = {"smoke": smoke, "decode": {}}
    for label, prompt in PROMPTS.items():
        samples = [request(prompt, 512), request(prompt, 512)]
        for sample in samples:
            message = sample["message"]
            combined = (message.get("content") or "") + (message.get("reasoning_content") or "")
            if sample["tokens"] != 512 or not combined.strip() or "\ufffd" in combined:
                raise RuntimeError(f"{label}: expected 512 valid generated tokens")
            if sample["graphs"] < 1 or sample["eager"]:
                raise RuntimeError(f"{label}: unexpected eager decode rounds")
        best = max(sample["tok_s"] for sample in samples)
        results["decode"][label] = {"spark_tok_s": SPARK_TOK_S[label], "sm120_tok_s": best,
                                    "speedup": best / SPARK_TOK_S[label], "samples": samples}
    document = " ".join(f"Entry {i}: the warehouse in sector {i % 37} holds {i * 7 % 911} crates."
                        for i in range(1400))
    results["prefill"] = {}
    for case, number, baseline in (("cold", 1203, 23.2), ("changed_tail", 77, 3.3)):
        prompt = f"{document}\n\nHow many crates does entry {number} hold? Answer with the number only."
        result = request(prompt, 200)
        expected = str(number * 7 % 911)
        answer = (result["message"].get("content") or "").strip().strip(".")
        results["prefill"][case] = {**result, "spark_wall_s": baseline,
                                     "speedup": baseline / result["wall_s"], "correct": answer == expected}
    passed = all(x["speedup"] >= 2 for x in results["decode"].values()) and all(
        x["speedup"] >= 2 and x["correct"] for x in results["prefill"].values()
    )
    results["passed_2x"] = passed
    (ROOT / "benchmark.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    for group in ("decode", "prefill"):
        for name, case in results[group].items():
            print(f"{group} {name}: {case['speedup']:.2f}x")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(benchmark())
