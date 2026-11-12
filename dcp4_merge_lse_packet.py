"""TensorFold's DCP LSE reduction with a contiguous packet per destination."""

import triton
import triton.language as tl


@triton.jit
def _merge_lse_packet(PO, PM, PL, OSEND, LSEND, R, HT: tl.constexpr, H: tl.constexpr,
                      LW: tl.constexpr, NCH: tl.constexpr):
    r = tl.program_id(0)
    h = tl.program_id(1)
    k = tl.arange(0, LW)
    m = float("-inf")
    l = 0.0
    o = tl.zeros((LW,), tl.float32)
    for c in range(NCH):
        base = (c * R + r) * HT + h
        cm = tl.load(PM + base)
        cl = tl.load(PL + base)
        co = tl.load(PO + base * LW + k)
        active = cl > 0.0
        next_m = tl.where(active, tl.maximum(m, cm), m)
        a = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
        b = tl.where(active, tl.exp(cm - next_m), 0.0)
        o = o * a + co * b
        l = l * a + cl * b
        m = next_m
    dst = h // H
    lane = r * H + h % H
    # Each packet starts with R*H*LW bf16 words, then R*H fp32 LSEs.
    o_stride = R * H * (LW + 2)
    l_stride = R * H * (LW // 2 + 1)
    has = l > 0.0
    tl.store(OSEND + dst * o_stride + lane * LW + k, tl.where(has, o / tl.where(has, l, 1.0), 0.0).to(tl.bfloat16))
    tl.store(LSEND + dst * l_stride + lane, tl.where(has, m + tl.log(tl.where(has, l, 1.0)), float("-inf")))


def check_layout():
    import torch
    from tensorfold.families.glm_moe_dsa.cuda.fused import _merge_lse

    torch.manual_seed(53)
    world, heads, width = 4, 16, 512
    for rows in (1, 2, 7, 16):
        for chunks in (1, 8):
            shape = (chunks, rows, world * heads)
            po = torch.randn((*shape, width), device="cuda")
            pm = torch.randn(shape, device="cuda")
            pl = torch.rand(shape, device="cuda")
            pl[:, :, 0] = 0
            head_rows = rows * heads
            n = head_rows * width
            output = torch.empty((world, n), device="cuda", dtype=torch.bfloat16)
            lse = torch.empty((world, head_rows), device="cuda")
            packet = torch.empty((world, n * 2 + head_rows * 4), device="cuda", dtype=torch.uint8)
            packet_output = packet.view(torch.bfloat16)
            packet_lse = packet.view(torch.float32)
            launch = (rows, world * heads)
            args = dict(HT=world * heads, H=heads, LW=width, NCH=chunks, num_warps=4)
            _merge_lse[launch](po, pm, pl, output, lse, rows, **args)
            _merge_lse_packet[launch](po, pm, pl, packet_output, packet_lse[0, n // 2:], rows, **args)
            assert torch.equal(output, packet_output[:, :n]), (rows, chunks, "output")
            assert torch.equal(lse, packet_lse[:, n // 2:]), (rows, chunks, "LSE")
    print("Packed per-destination partials and LSE match the original kernel bit-for-bit.")


if __name__ == "__main__":
    check_layout()
