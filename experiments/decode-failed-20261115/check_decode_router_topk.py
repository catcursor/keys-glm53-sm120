#!/usr/bin/env python3
"""Bitwise and graph-free timing checks against the actual installed reference.

PYTHONPATH must include this directory.
Example: python3 check_decode_router_topk.py --config /model/config.json \
    --reference /port/python-reference/src/tensorfold/families/glm5_next/cuda/glue.py --out check.json
This measures router finalization only; unchanged _router_part is excluded.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import statistics
import struct
import sys
import time

import torch
import triton
from decode_router_topk import _decode_router_topk


def shape(config_path):
    doc = json.loads(Path(config_path).read_text())
    c = doc.get("text_config", doc)
    assert c["model_type"] == "glm_moe_dsa"
    assert c.get("n_group", 1) == c.get("topk_group", 1) == 1
    # Zig supplies routed_scaling as a runtime f32, not a Python fp64 constexpr.
    scale = struct.unpack("<f", struct.pack("<f", c["routed_scaling_factor"]))[0]
    return int(c["n_routed_experts"]), int(c["num_experts_per_tok"]), scale, bool(c["norm_topk_prob"])


def inputs(rows, ne, kind, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    part = torch.randn((8, rows, ne), generator=gen, device="cuda", dtype=torch.float32)
    bias = torch.randn((ne,), generator=gen, device="cuda", dtype=torch.float32) * 0.05
    if kind == "ties":
        part.zero_()
        bias.zero_()
    elif kind == "cancellation":
        part.zero_()
        part[0].fill_(2.0 ** 24)
        part[1] = torch.randn((rows, ne), generator=gen, device="cuda")
        part[2].fill_(-(2.0 ** 24))
        part[3] = torch.randn((rows, ne), generator=gen, device="cuda") * (2.0 ** -15)
        part[4].fill_(2.0 ** 20)
        part[5].fill_(-(2.0 ** 20))
        bias.zero_()
    elif kind == "near_ties":
        part.mul_(2.0 ** -22)
        bias.zero_()
    elif kind == "saturated":
        part.zero_()
        part[0] = torch.linspace(-100.0, 100.0, ne, device="cuda").expand(rows, ne)
        bias.zero_()
    elif kind != "random":
        raise ValueError(kind)
    return part, bias


def outputs(rows, ne, k):
    return (torch.empty((rows, ne), device="cuda", dtype=torch.float32),
            torch.empty((rows, k), device="cuda", dtype=torch.int32),
            torch.empty((rows, k), device="cuda", dtype=torch.float32))


def launchers(part, bias, k, scale, norm, old, new):
    _, rows, ne = part.shape
    total = rows * ne
    slots = old[1].shape[1]
    assert new[1].shape[1] == slots and slots in (k, k + 1)
    params = dict(NE=ne, TOPK=k, SLOTS=slots, BLOCK=triton.next_power_of_2(ne + 1),
                  SLOTP=triton.next_power_of_2(k + 1), NORM=norm, num_warps=4)

    def reference():
        glue._router_sum[(triton.cdiv(total, 1024),)](
            part, old[0], total, KS=8, BLOCK=1024, num_warps=4)
        glue._topk[(rows,)](old[0], bias, old[1], old[2], scale, **params)

    def candidate():
        _decode_router_topk[(rows,)](
            part, new[0], bias, new[1], new[2], total, scale, KS=8, **params)

    return reference, candidate


def check(old, new, label):
    for name, a, b in zip(("logits", "pick", "wts"), old, new):
        # Integer views compare signed zeros and all FP32 bits, unlike allclose.
        ai, bi = a.view(torch.int32), b.view(torch.int32)
        if not torch.equal(ai, bi):
            mismatches = torch.nonzero(ai != bi)
            first = mismatches[0].tolist()
            idx = tuple(first)
            raise AssertionError(f"{label} {name}: {mismatches.shape[0]} bit mismatches; "
                                 f"first {first}: {a[idx].item()} vs {b[idx].item()}")


def timed(fn, iterations):
    # Ordinary launches only. Events include any host enqueue gaps; wall time also
    # includes the final wait. Neither metric is a CUDA Graph replay measurement.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    wall_us = (time.perf_counter_ns() - t0) / (1000 * iterations)
    timeline_us = start.elapsed_time(end) * 1000 / iterations
    return {"wall_us": wall_us, "event_timeline_us": timeline_us}


def main():
    global glue
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--reference", required=True,
                    help="Exact production glm5_next/cuda/glue.py file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    assert args.iterations > 0 and args.repeats > 0 and args.seeds > 0
    spec = importlib.util.spec_from_file_location("decode_router_reference_glue", args.reference)
    assert spec is not None and spec.loader is not None
    glue = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = glue
    spec.loader.exec_module(glue)
    ne, k, scale, configured_norm = shape(args.config)
    assert 0 < k <= ne
    ref_path = Path(inspect.getfile(glue))
    report = {"gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
              "torch": torch.__version__, "triton": triton.__version__, "graphs": False,
              "reference": str(ref_path), "reference_sha256": hashlib.sha256(ref_path.read_bytes()).hexdigest(),
              "router_sum_source_sha256": hashlib.sha256(glue._router_sum.src.encode()).hexdigest(),
              "topk_source_sha256": hashlib.sha256(glue._topk.src.encode()).hexdigest(),
              "shape": {"experts": ne, "topk": k, "scale": scale, "norm": configured_norm, "ks": 8,
                        "native_slots": k, "tested_slots": [k, k + 1]},
              "checks": [], "timings": []}
    for rows in (1, 2, 4, 8, 16, 32):
        for kind in ("random", "ties", "cancellation", "near_ties", "saturated"):
            for seed in range(args.seeds):
                part, bias = inputs(rows, ne, kind, seed)
                for slots in (k, k + 1):
                    old, new = outputs(rows, ne, slots), outputs(rows, ne, slots)
                    for norm in (configured_norm, not configured_norm):
                        baseline, fused = launchers(part, bias, k, scale, norm, old, new)
                        baseline()
                        fused()
                        check(old, new, f"rows={rows} case={kind} seed={seed} norm={norm} slots={slots}")
                        if kind == "ties":
                            expected = torch.arange(k, device="cuda", dtype=torch.int32)
                            if slots == k + 1:
                                expected = torch.cat((expected, torch.tensor([ne], device="cuda", dtype=torch.int32)))
                            assert torch.equal(new[1], expected.expand(rows, slots)), "lower-id tie break changed"
                        report["checks"].append({"rows": rows, "case": kind, "seed": seed,
                                                 "norm": norm, "slots": slots, "bitwise": True})
        part, bias = inputs(rows, ne, "random", 12345)
        old, new = outputs(rows, ne, k), outputs(rows, ne, k)
        baseline, fused = launchers(part, bias, k, scale, configured_norm, old, new)
        for _ in range(20):
            baseline()
            fused()
        pairs = []
        for repeat in range(args.repeats):
            pair = {}
            order = (("baseline", baseline), ("fused", fused))
            if repeat % 2:
                order = order[::-1]
            for name, fn in order:
                pair[name] = timed(fn, args.iterations)
            pairs.append(pair)
        timing = {"rows": rows, "samples": pairs}
        for name in ("baseline", "fused"):
            timing[name] = {metric: statistics.median(x[name][metric] for x in pairs)
                            for metric in ("wall_us", "event_timeline_us")}
        report["timings"].append(timing)
        print(json.dumps({"rows": rows, "bitwise": True,
                          "baseline": timing["baseline"], "fused": timing["fused"]}), flush=True)
    report["passed"] = True
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
