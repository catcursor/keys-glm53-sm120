"""Score adjacent GLM-5.3 prefill queries against a shared index key tile."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T", "R0", "N"])
def _index_scores_pair(Q, W, IC, POS, OUT, T, R0, N, NH: tl.constexpr, D: tl.constexpr,
                       BTT: tl.constexpr, WSCALE: tl.constexpr, QSCALE: tl.constexpr):
    group = tl.program_id(0) * 2
    t0 = tl.program_id(1) * BTT
    row = group + tl.arange(0, 2)
    pos = tl.load(POS) + R0
    t = t0 + tl.arange(0, BTT)
    valid = (row[:, None] < N) & (t[None, :] < T)
    causal = t[None, :] <= pos + row[:, None]
    if t0 <= pos + tl.minimum(group + 2, N) - 1:
        qh = tl.arange(0, 2 * NH)
        d = tl.arange(0, D)
        q = tl.load(Q + (group * NH + qh[:, None]) * D + d[None, :],
                    mask=group + qh[:, None] // NH < N, other=0)
        key_valid = (t < T) & (t <= pos + tl.minimum(group + 2, N) - 1)
        k = tl.load(IC + t[:, None].to(tl.int64) * D + d[None, :], mask=key_valid[:, None], other=0)
        dot = tl.maximum(tl.dot(q, tl.trans(k)) * QSCALE, 0.0)
        weight = tl.load(W + group * NH + qh, mask=group + qh // NH < N, other=0) * WSCALE
        scores = tl.sum(tl.reshape(weight[:, None] * dot, (2, NH, BTT)), 1)
        bits = scores.to(tl.int32, bitcast=True)
        key = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
        order = tl.where(causal, key ^ -2147483648, 0)
    else:
        order = tl.full((2, BTT), 0, tl.int32)
    tl.store(OUT + row[:, None].to(tl.int64) * T + t[None, :], order, mask=valid)
