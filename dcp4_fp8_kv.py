"""TensorFold SM120 DCP4 MLA cache kernels.

A row is 512 E4M3FN bytes, four FP32 scales, and 64 BF16 RoPE values.
"""

import triton
import triton.language as tl
from tensorfold.families.glm_moe_dsa.cuda.fused import _rope_pair


@triton.jit
def _store_row(LC, row, latent, rope_lo, rope_hi):
    groups = tl.reshape(latent.to(tl.float32), (4, 128))
    scale = tl.maximum(tl.max(tl.abs(groups), 1), 1.0e-10) / 448.0
    enc = tl.clamp(groups / scale[:, None], -448.0, 448.0).to(tl.float8e4nv)
    fp8 = tl.cast(LC, tl.pointer_type(tl.float8e4nv))
    f32 = tl.cast(LC, tl.pointer_type(tl.float32))
    bf16 = tl.cast(LC, tl.pointer_type(tl.bfloat16))
    tl.store(fp8 + row * 656 + tl.arange(0, 512), tl.reshape(enc, (512,)))
    tl.store(f32 + row * 164 + 128 + tl.arange(0, 4), scale)
    lane = tl.arange(0, 32)
    tl.store(bf16 + row * 328 + 264 + lane, rope_lo.to(tl.bfloat16))
    tl.store(bf16 + row * 328 + 296 + lane, rope_hi.to(tl.bfloat16))


@triton.jit
def _kv_write_fp8(KVA, kva_stride, NW, LC, POS, INV, eps, DCP: tl.constexpr, RANK: tl.constexpr):
    _kv_write_fp8_impl(KVA, kva_stride, NW, LC, POS, None, INV, eps, DCP, RANK, False)


@triton.jit
def _kv_write_fp8_rows(KVA, kva_stride, NW, LC, POS, BASE, INV, eps, DCP: tl.constexpr, RANK: tl.constexpr):
    _kv_write_fp8_impl(KVA, kva_stride, NW, LC, POS, BASE, INV, eps, DCP, RANK, True)


@triton.jit
def _kv_write_fp8_impl(KVA, kva_stride, NW, LC, POS, BASE, INV, eps,
                       DCP: tl.constexpr, RANK: tl.constexpr, ROWS: tl.constexpr):
    r = tl.program_id(0)
    if ROWS:
        pg = tl.load(POS + r).to(tl.int64)
        slot = tl.load(BASE + r).to(tl.int64)
    else:
        pg = (tl.load(POS) + r).to(tl.int64)
        slot = 0
    if pg % DCP != RANK:
        return
    k = tl.arange(0, 512)
    x = tl.load(KVA + r * kva_stride + k).to(tl.float32)
    rinv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / 512.0 + eps)
    w = tl.load(NW + k).to(tl.float32)
    latent = (w * (x * rinv).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    i = tl.arange(0, 32)
    a = tl.load(KVA + r * kva_stride + 512 + 2 * i).to(tl.float32)
    b = tl.load(KVA + r * kva_stride + 513 + 2 * i).to(tl.float32)
    angle = pg.to(tl.float32) * tl.load(INV + i)
    lo, hi = _rope_pair(a, b, tl.cos(angle), tl.sin(angle))
    _store_row(LC, slot + pg // DCP, latent, lo, hi)


@triton.jit
def _pack_kv_fp8(SRC, LC, STRIDE: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    src = SRC + r * STRIDE
    latent = tl.load(src + tl.arange(0, 512))
    lo = tl.load(src + 512 + tl.arange(0, 32))
    hi = tl.load(src + 544 + tl.arange(0, 32))
    _store_row(LC, r, latent, lo, hi)


@triton.jit
def _load_rows(LC, rows, valid, k, rope):
    f32 = tl.cast(LC, tl.pointer_type(tl.float32))
    bf16 = tl.cast(LC, tl.pointer_type(tl.bfloat16))
    raw = tl.load(LC + rows[:, None] * 656 + k[None, :], valid[:, None], 0)
    quantized = raw.to(tl.float8e4nv, bitcast=True)
    scale = tl.load(f32 + rows[:, None] * 164 + 128 + (k[None, :] // 128), valid[:, None], 0)
    latent = (quantized.to(tl.float32) * scale).to(tl.bfloat16)
    key = tl.load(bf16 + rows[:, None] * 328 + 264 + rope[None, :], valid[:, None], 0)
    return latent, key


@triton.jit
def _unpack_kv_fp8(LC, DST, ROWS, PAD, DCP: tl.constexpr):
    r = (tl.program_id(0) * 8 + tl.arange(0, 8)).to(tl.int64)
    valid = r < ROWS
    shard_row = (r % DCP) * PAD + r // DCP
    k = tl.arange(0, 512)
    rope = tl.arange(0, 64)
    latent, key = _load_rows(LC, shard_row, valid, k, rope)
    tl.store(DST + r[:, None] * 576 + k[None, :], latent, valid[:, None])
    tl.store(DST + r[:, None] * 576 + 512 + rope[None, :], key, valid[:, None])


@triton.jit
def _attn_dcp_fp8(QALL, LC, TOK, CNT, POS, PO, PM, PL, R, H: tl.constexpr,
                  G: tl.constexpr, LW: tl.constexpr, RD: tl.constexpr, K: tl.constexpr,
                  CHK: tl.constexpr, KTT: tl.constexpr, SCALE: tl.constexpr,
                  DCP: tl.constexpr, RANK: tl.constexpr):
    _attn_dcp_fp8_impl(QALL, LC, TOK, CNT, POS, None, PO, PM, PL, R, H, G, LW, RD,
                       K, CHK, KTT, SCALE, DCP, RANK, False)


@triton.jit
def _attn_dcp_fp8_rows(QALL, LC, TOK, CNT, POS, BASE, PO, PM, PL, R, H: tl.constexpr,
                       G: tl.constexpr, LW: tl.constexpr, RD: tl.constexpr, K: tl.constexpr,
                       CHK: tl.constexpr, KTT: tl.constexpr, SCALE: tl.constexpr,
                       DCP: tl.constexpr, RANK: tl.constexpr):
    _attn_dcp_fp8_impl(QALL, LC, TOK, CNT, POS, BASE, PO, PM, PL, R, H, G, LW, RD,
                       K, CHK, KTT, SCALE, DCP, RANK, True)


@triton.jit
def _attn_dcp_fp8_impl(QALL, LC, TOK, CNT, POS, BASE, PO, PM, PL, R, H: tl.constexpr,
                       G: tl.constexpr, LW: tl.constexpr, RD: tl.constexpr, K: tl.constexpr,
                       CHK: tl.constexpr, KTT: tl.constexpr, SCALE: tl.constexpr,
                       DCP: tl.constexpr, RANK: tl.constexpr, ROWS: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1)
    grp = tl.program_id(2)
    if ROWS:
        p = tl.load(POS + r)
        slot = tl.load(BASE + r).to(tl.int64)
    else:
        p = tl.load(POS) + r
        slot = 0
    if p < K:
        n = tl.where(p >= RANK, (p - RANK) // DCP + 1, 0)
    else:
        n = tl.load(CNT + r)
    hh = tl.arange(0, H)
    kl = tl.arange(0, LW)
    kr = tl.arange(0, RD)
    m = tl.full((H,), float("-inf"), tl.float32)
    l = tl.zeros((H,), tl.float32)
    o = tl.zeros((H, LW), tl.float32)
    start = c * CHK
    if start < n:
        qb = QALL + ((grp * R + r) * H + hh[:, None]) * (LW + RD)
        ql = tl.load(qb + kl[None, :])
        qr = tl.load(qb + LW + kr[None, :])
        for t in range(CHK // KTT):
            idx = start + t * KTT + tl.arange(0, KTT)
            ok = idx < n
            if p < K:
                row = idx.to(tl.int64)
            else:
                row = tl.load(TOK + r * K + idx, mask=ok, other=0).to(tl.int64)
            kv, kro = _load_rows(LC, slot + row, ok, kl, kr)
            s = (tl.dot(ql, tl.trans(kv)) + tl.dot(qr, tl.trans(kro))) * SCALE
            s = tl.where(ok[None, :], s, float("-inf"))
            tile_m = tl.max(s, 1)
            active = tile_m != float("-inf")
            next_m = tl.where(active, tl.maximum(m, tile_m), m)
            alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
            pr = tl.where(ok[None, :] & active[:, None], tl.exp(s - next_m[:, None]), 0.0)
            o = o * alpha[:, None] + tl.dot(pr.to(tl.bfloat16), kv)
            l = l * alpha + tl.sum(pr, 1)
            m = next_m
    base = (c * R + r) * (G * H) + grp * H + hh
    tl.store(PO + base[:, None] * LW + kl[None, :], o)
    tl.store(PM + base, m)
    tl.store(PL + base, l)
