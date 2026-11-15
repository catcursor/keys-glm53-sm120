"""Decode router finalization, preserving the reference FP32 slice and pick order.

The tensor-core router partials are unchanged. This replaces only _router_sum
followed by _topk from tensorfold.families.glm5_next.cuda.glue. Logits are still
stored, so callers and diagnostics see all three reference outputs.
"""

import triton
import triton.language as tl


@triton.jit
def _decode_router_topk(PART, L, BIAS, PICK, WTS, total, scale,
                        KS: tl.constexpr, NE: tl.constexpr, TOPK: tl.constexpr,
                        SLOTS: tl.constexpr, BLOCK: tl.constexpr,
                        SLOTP: tl.constexpr, NORM: tl.constexpr):
    r = tl.program_id(0)
    ar = tl.arange(0, BLOCK)
    ak = tl.arange(0, SLOTP)
    ok = ar < NE
    i = r * NE + ar
    # Exactly _router_sum: initialize from slice 0, then add slices 1..KS-1.
    # No cross-slice tl.sum, new dot, or change in accumulator precision.
    lg = tl.load(PART + i, mask=ok, other=0.0)
    for s in tl.static_range(1, KS):
        lg = lg + tl.load(PART + s * total + i, mask=ok, other=0.0)
    tl.store(L + i, lg, mask=ok)
    # Keep the original _topk vector size, warps, tie-break and weight reduction.
    score = 1.0 / (1.0 + tl.exp(-lg))
    choice = tl.where(ok, score + tl.load(BIAS + ar, mask=ok, other=0.0), float("-inf"))
    picks = tl.zeros((SLOTP,), dtype=tl.int32)
    wts = tl.zeros((SLOTP,), dtype=tl.float32)
    weight_total = 0.0
    for k in tl.static_range(TOPK):
        mx = tl.max(choice, axis=0)
        idx = tl.min(tl.where(choice == mx, ar, BLOCK), axis=0)
        sk = tl.sum(tl.where(ar == idx, score, 0.0), axis=0)
        picks = tl.where(ak == k, idx, picks)
        wts = tl.where(ak == k, sk, wts)
        weight_total += sk
        choice = tl.where(ar == idx, float("-inf"), choice)
    if NORM:
        wts = wts / (weight_total + 1e-20)
    wts = wts * scale
    picks = tl.where(ak == TOPK, NE, picks)
    wts = tl.where(ak == TOPK, 1.0, wts)
    tl.store(PICK + r * SLOTS + ak, picks, mask=ak < SLOTS)
    tl.store(WTS + r * SLOTS + ak, wts, mask=ak < SLOTS)
