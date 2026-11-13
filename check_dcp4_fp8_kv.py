"""Check the FP8 shard format, DCP ownership, and attention against BF16."""
import math

import torch
import triton
from tensorfold.families.glm_moe_dsa.cuda.fused import _attn_dcp, _kv_write

from dcp4_fp8_kv import (
    _attn_dcp_fp8,
    _attn_dcp_fp8_rows,
    _kv_write_fp8,
    _kv_write_fp8_rows,
    _kv_sparse_mark,
    _kv_sparse_offsets,
    _kv_sparse_pack,
    _kv_sparse_scan,
    _kv_sparse_unpack,
    _pack_kv_fp8,
    _unpack_kv_fp8,
)


def check_sparse_gather():
    world, end, rows, width, block = 4, 65537, 22, 576, 256
    padded = triton.cdiv(end, world)
    tiles = triton.cdiv(padded, block)
    device = "cuda"
    source = torch.randn((end + rows, width), dtype=torch.bfloat16, device=device)
    shards = torch.empty((world, padded, 656), dtype=torch.uint8, device=device)
    for rank in range(world):
        _pack_kv_fp8[(len(source[rank:end:world]),)](source[rank:], shards[rank], world * width)
    reference = source.clone()
    _unpack_kv_fp8[(triton.cdiv(end, 8),)](shards, reference, end, padded, world)

    candidates = torch.randint(end // world, (8_192,), device=device) * world
    candidates[::3] += 2
    candidates[0] = end - 1
    tokens = candidates[torch.randint(len(candidates), (rows, 2048), device=device)].to(torch.int32)
    tokens[:, -1] = end + torch.arange(rows, dtype=torch.int32, device=device)
    selected = tokens[tokens < end].unique(sorted=True).long()
    by_rank = [selected[selected % world == rank] // world for rank in range(world)]
    pad = max(map(len, by_rank))
    plan = torch.zeros(end, dtype=torch.int32, device=device)
    offsets = torch.empty((world, tiles), dtype=torch.int32, device=device)
    counts = torch.empty(world, dtype=torch.int32, device=device)
    compact = torch.zeros((world, pad, 656), dtype=torch.uint8, device=device)
    actual = source.clone()

    def gather():
        plan.zero_()
        _kv_sparse_mark[(triton.cdiv(tokens.numel(), block),)](tokens, plan, tokens.numel(), end, block)
        _kv_sparse_scan[(tiles, world)](plan, offsets, end, tiles, world, block)
        _kv_sparse_offsets[(world,)](offsets, counts, tiles, 2048)
        for rank in range(world):
            _kv_sparse_pack[(triton.cdiv(padded, 8),)](
                shards[rank], compact[rank], plan, offsets, end, tiles, rank, world, block)
        _kv_sparse_unpack[(triton.cdiv(end, 8),)](
            compact, actual, plan, offsets, end, pad, tiles, world, block)

    gather()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gather()
    graph.replay()
    assert counts.tolist() == list(map(len, by_rank))
    for rank, local in enumerate(by_rank):
        torch.testing.assert_close(compact[rank, :len(local)], shards[rank, local], rtol=0, atol=0)
    expected = source.clone()
    expected[selected] = reference[selected]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print("Sparse FP8 gather restored the selected historical rows bit-for-bit and left all other rows intact.")


def check_slot_offsets():
    world, streams, local = 4, 4, 576
    device = "cuda"
    positions = torch.tensor([5, 18, 2063, 2208], dtype=torch.int32, device=device)
    bases = torch.arange(streams, dtype=torch.int32, device=device) * local
    kva = torch.randn((streams, 640), dtype=torch.bfloat16, device=device)
    nw = torch.ones(512, dtype=torch.bfloat16, device=device)
    inv = torch.logspace(0, -5, 32, device=device)
    expected = torch.zeros((world, streams, local, 656), dtype=torch.uint8, device=device)
    batched = torch.zeros_like(expected)

    for rank in range(world):
        for s in range(streams):
            _kv_write_fp8[(1,)](kva[s], 640, nw, expected[rank, s], positions[s:], inv,
                                 1e-5, world, rank)

    def write_slots():
        for rank in range(world):
            _kv_write_fp8_rows[(streams,)](kva, 640, nw, batched[rank], positions, bases,
                                           inv, 1e-5, world, rank)

    write_slots()
    batched.zero_()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write_slots()
    graph.replay()
    torch.testing.assert_close(batched, expected, atol=0, rtol=0)

    heads, nch, topk = 16, 8, 2048
    keys = torch.randn((streams, local, 576), dtype=torch.bfloat16, device=device)
    for s in range(streams):
        _pack_kv_fp8[(local,)](keys[s], batched[0, s], 576)
    queries = torch.randn((world, streams, heads, 576), dtype=torch.bfloat16, device=device)
    ids = torch.arange(topk, dtype=torch.int32, device=device).repeat(streams, 1) % local
    counts = torch.tensor([0, 0, 390, 519], dtype=torch.int32, device=device)

    def outputs(rows):
        partial = torch.empty((nch, rows, world * heads), device=device)
        return torch.empty((*partial.shape, 512), device=device), partial, torch.empty_like(partial)

    config = (heads, world, 512, 64, topk, 256, 32, 1.0 / math.sqrt(256), world, 0)
    actual = outputs(streams)

    def read_slots():
        _attn_dcp_fp8_rows[(streams, nch, world)](queries, batched[0], ids, counts, positions,
                                                 bases, *actual, streams, *config,
                                                 num_warps=4, num_stages=1)

    read_slots()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        read_slots()
    graph.replay()
    for s in range(streams):
        reference = outputs(1)
        _attn_dcp_fp8[(1, nch, world)](queries[:, s:s + 1].contiguous(), batched[0, s],
                                       ids[s:], counts[s:], positions[s:], *reference, 1, *config,
                                       num_warps=4, num_stages=1)
        for got, want in zip(actual, reference, strict=True):
            torch.testing.assert_close(got[:, s:s + 1], want, atol=1e-5, rtol=1e-5)
    print("Four FP8 slots: CUDA Graph writes and mixed short/sparse attention agree with isolated slots.")


def main():
    torch.manual_seed(23)
    device = "cuda"
    world, n = 4, 617
    pad = triton.cdiv(n, world)
    rows = torch.randn((n, 576), dtype=torch.bfloat16, device=device)
    shards = torch.zeros((world, pad, 656), dtype=torch.uint8, device=device)
    restored = torch.empty_like(rows)

    def pack():
        for rank in range(world):
            _pack_kv_fp8[(len(rows[rank::world]),)](rows[rank:], shards[rank], 576 * world)
        _unpack_kv_fp8[(triton.cdiv(n, 8),)](shards, restored, n, pad, world)

    pack()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pack()
    graph.replay()
    torch.testing.assert_close(restored[:, 512:], rows[:, 512:], atol=0, rtol=0)
    delta = restored[:, :512].float() - rows[:, :512].float()
    rms_relative = (delta.square().mean() / rows[:, :512].float().square().mean()).sqrt().item()
    assert rms_relative < 0.03, rms_relative

    kva = torch.randn((9, 640), dtype=torch.bfloat16, device=device)
    nw = torch.ones(512, dtype=torch.bfloat16, device=device)
    inv = torch.logspace(0, -5, 32, device=device)
    pos = torch.tensor([5], dtype=torch.int32, device=device)
    eps = 1.0e-5
    for rank in range(world):
        ref = torch.zeros((4, 576), dtype=torch.bfloat16, device=device)
        expected = torch.empty((4, 656), dtype=torch.uint8, device=device)
        actual = torch.zeros_like(expected)
        _kv_write[(9,)](kva, 640, nw, ref, pos, inv, eps, 512, 64, world, rank)
        _pack_kv_fp8[(4,)](ref, expected, 576)
        _kv_write_fp8[(9,)](kva, 640, nw, actual, pos, inv, eps, world, rank)
        owned = [i // world for i in range(5, 14) if i % world == rank]
        torch.testing.assert_close(actual[owned], expected[owned], atol=0, rtol=0)

    r, heads, nch = 2, 16, 8
    queries = torch.randn((world, r, heads, 576), dtype=torch.bfloat16, device=device)
    ids = torch.arange(2048, dtype=torch.int32, device=device).repeat(r, 1) % (n // world)
    counts = torch.tensor([136, 151], dtype=torch.int32, device=device)
    pos.fill_(4096)

    def outputs():
        m = torch.empty((nch, r, world * heads), device=device)
        return torch.empty((*m.shape, 512), device=device), m, torch.empty_like(m)

    a, b = outputs(), outputs()
    config = (r, heads, world, 512, 64, 2048, 256, 32, 1.0 / math.sqrt(256), world, 0)
    _attn_dcp[(r, nch, world)](queries, restored[::world].contiguous(), ids, counts, pos, *a, *config,
                               num_warps=4, num_stages=1)
    _attn_dcp_fp8[(r, nch, world)](queries, shards[0], ids, counts, pos, *b, *config,
                                   num_warps=4, num_stages=1)
    torch.testing.assert_close(b[1], a[1], atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(b[2], a[2], atol=2.0e-4, rtol=2.0e-4)
    attention_rms = ((b[0] - a[0]).square().mean() / a[0].square().mean()).sqrt().item()
    assert attention_rms < 1.0e-3, attention_rms
    print(f"CUDA Graph pack/unpack passed; RoPE bit exact; latent relative RMS error {rms_relative:.4%}")
    print(f"All DCP owners wrote expected bytes; sparse attention relative RMS difference {attention_rms:.4%}")
    check_slot_offsets()
    check_sparse_gather()


if __name__ == "__main__":
    main()
