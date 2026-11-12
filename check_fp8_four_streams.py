"""Verify four overlapping FP8 requests, with isolated answers and graph replay."""

import argparse
import concurrent.futures
import json
import re
import threading
import time
import urllib.request
from pathlib import Path


def make_prompt(marker, entries):
    document = " ".join(f"Inventory item {i} is in bin {i % 71}." for i in range(entries))
    return (f"The verification code for this request is {marker}.\n{document}\n"
            "Start the answer with the exact verification code. Then print the integers "
            "1 through 150, separated by spaces. Do not omit any integers.")


def make_request(base, marker, entries, *, stream=False):
    body = {
        "model": "model",
        "messages": [{"role": "user", "content": make_prompt(marker, entries)}],
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 512,
        "stream": stream,
    }
    return urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                  {"Content-Type": "application/json"})


def request(base, barrier, marker, entries):
    query = make_request(base, marker, entries)
    barrier.wait(timeout=15)
    with urllib.request.urlopen(query, timeout=180) as response:
        result = json.load(response)
    return result["usage"], result["choices"][0]["message"].get("content") or ""


def check_cross_slot_reuse(base, log, require_profile):
    marker = "VIOLET-8042"
    offset = log.stat().st_size
    with urllib.request.urlopen(make_request(base, marker, 250, stream=True), timeout=180) as first:
        for line in first:
            choices = json.loads(line[6:]).get("choices", []) if line.startswith(b"data: {") else []
            if choices and choices[0]["delta"].get("content"):
                break
        else:
            raise RuntimeError("Source stream never started decoding")
        with urllib.request.urlopen(make_request(base, marker, 250), timeout=180) as follower:
            replay = json.load(follower)
        first.read()
    if marker not in (replay["choices"][0]["message"].get("content") or ""):
        raise RuntimeError("Cross-slot replay returned the wrong answer")
    with log.open("rb") as stream:
        stream.seek(offset)
        observed = stream.read().decode(errors="replace")
    rows = re.findall(r"done req-\d+ prompt=(\d+) cached=(\d+).*?graphs=(\d+) eager=(\d+).*? sid=(\d+)", observed)
    if len(rows) != 2 or sum(int(cached) == int(prompt) for prompt, cached, *_ in rows) != 1:
        raise RuntimeError(f"Expected one complete prefix replay, observed {rows}")
    if require_profile and not re.search(r"glm53 prof rank 0: rounds=\d+ streams=2\b", observed):
        raise RuntimeError("The replay did not overlap the source stream")
    return int(replay["usage"]["prompt_tokens"])


def post(base, path, body, timeout=10):
    request = urllib.request.Request(base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def metric(base, name):
    with urllib.request.urlopen(base + "/metrics", timeout=3) as response:
        key = ("tensorfold:" + name).encode()
        for line in response.read().splitlines():
            if line.startswith(key + b" "):
                return int(line.split()[1])
    raise RuntimeError(f"Missing metric {name}")


def check_shared_budget(base, context):
    marker = "PLATINUM-7215"
    text = (" Archived inventory entry." * 48750
            + f"\nThe verification code is {marker}. Reply with that code only.")
    long_request = {
        "model": "model",
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    tokenized = post(base, "/tokenize", long_request, timeout=30)
    prompt_tokens = tokenized["count"]
    if tokenized["max_model_len"] != context or not context // 4 < prompt_tokens < context:
        raise RuntimeError(f"Unexpected shared-context check size {prompt_tokens}")
    # Reserve all of the configured pool. The answer should stop naturally long before the limit.
    long_request["max_tokens"] = context - prompt_tokens
    short_request = dict(long_request, max_tokens=128,
                         messages=[{"role": "user", "content": "Reply with the code OPAL-6201 only."}])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        large = pool.submit(post, base, "/v1/chat/completions", long_request, 600)
        for _ in range(50):
            if large.done() or metric(base, "requests_running") > 0:
                break
            time.sleep(0.1)
        small = pool.submit(post, base, "/v1/chat/completions", short_request, 600)
        queued = False
        for _ in range(50):
            if large.done():
                break
            if metric(base, "requests_waiting") > 0:
                queued = True
                break
            time.sleep(0.1)
        first, second = large.result(), small.result()
    if not queued or first["usage"]["prompt_tokens"] != prompt_tokens:
        raise RuntimeError("The full-pool reservation did not queue the smaller request")
    for reply, expected in ((first, marker), (second, "OPAL-6201")):
        if expected not in (reply["choices"][0]["message"].get("content") or ""):
            raise RuntimeError(f"Expected {expected} after reserving the shared KV pool")
    return {"prompt_tokens": prompt_tokens, "reserved_tokens": context, "queued_then_admitted": queued}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18992)
    parser.add_argument("--rank-log", type=Path, required=True)
    parser.add_argument("--no-round-profile", action="store_true",
                        help="Check API concurrency without requiring detailed GPU profiling.")
    parser.add_argument("--shared-context", type=int, help="Also test long input and full-pool admission.")
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    log = args.rank_log
    startup = log.read_text(errors="replace")
    if not re.search(r"[1-9]\d* concurrent decode graphs captured", startup):
        raise RuntimeError("Four-stream CUDA Graph capture has not completed")
    offset = log.stat().st_size
    markers = ("AMBER-2137", "COBALT-4651", "JADE-6283", "SILVER-9539")
    barrier = threading.Barrier(len(markers))
    peak_running = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(markers)) as pool:
        futures = [pool.submit(request, base, barrier, marker, n)
                   for marker, n in zip(markers, (2, 18, 330, 480), strict=True)]
        while peak_running < 4 and not all(future.done() for future in futures):
            peak_running = max(peak_running, metric(base, "requests_running"))
            time.sleep(0.1)
        results = [future.result() for future in futures]
    for marker, (usage, answer) in zip(markers, results, strict=True):
        if marker not in answer or any(other in answer for other in markers if other != marker):
            raise RuntimeError(f"Wrong or cross-stream answer for {marker}: {answer[:200]!r}")
        if usage["completion_tokens"] < 24:
            raise RuntimeError(f"Answer for {marker} was too short to exercise overlap")
    with log.open("rb") as stream:
        stream.seek(offset)
        observed = stream.read().decode(errors="replace")
    completions = re.findall(r"done req-\d+ .*?graphs=(\d+) eager=(\d+).*? sid=(\d+)", observed)
    if len(completions) != 4 or any(int(graphs) == 0 or int(eager) for graphs, eager, _ in completions):
        raise RuntimeError(f"Expected four graph-only completions, observed {completions}")
    four = re.findall(r"glm53 prof rank 0: rounds=(\d+) streams=4\b[^\n]*rows=4\.00", observed)
    if not args.no_round_profile and not four:
        raise RuntimeError("No four-stream CUDA Graph round in the runtime profile")
    if peak_running < 4 and not four:
        raise RuntimeError(f"Only observed {peak_running} active requests")
    result = {"requests": [usage for usage, _ in results], "graphs_eager_sid": completions,
              "peak_running": peak_running, "four_stream_rounds": sum(map(int, four)),
              "cross_slot_cached_tokens": check_cross_slot_reuse(base, log, not args.no_round_profile)}
    if args.shared_context:
        result["shared_budget"] = check_shared_budget(base, args.shared_context)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
