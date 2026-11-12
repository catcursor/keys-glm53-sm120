"""Check the port's local NCCL collectives and their CUDA Graph capture."""

import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def check_rank(rank):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=4, device_id=device,
                            timeout=timedelta(seconds=60))
    try:
        for count in (6144, 8 * 6144):
            local = torch.full((count,), rank + 1, dtype=torch.float32, device=device)
            gathered = torch.empty(4 * count, dtype=local.dtype, device=device)
            dist.all_gather_into_tensor(gathered, local)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                dist.all_gather_into_tensor(gathered, local)
            graph.replay()
            torch.cuda.synchronize(device)
            expected = torch.arange(1, 5, dtype=local.dtype, device=device)[:, None].expand(4, count)
            torch.testing.assert_close(gathered.view(4, count), expected, rtol=0, atol=0)
            start = torch.cuda.Event(enable_timing=True, external=True)
            finish = torch.cuda.Event(enable_timing=True, external=True)
            rounds = 128
            batched = torch.cuda.CUDAGraph()
            dist.barrier(device_ids=[rank])
            torch.cuda.synchronize(device)
            with torch.cuda.graph(batched):
                start.record()
                for _ in range(rounds):
                    dist.all_gather_into_tensor(gathered, local)
                finish.record()
            batched.replay()
            finish.synchronize()
            print(json.dumps({"rank": rank, "bytes_per_rank": local.nbytes,
                              "mean_us": start.elapsed_time(finish) * 1000 / rounds}), flush=True)
            torch.cuda.synchronize(device)
            graph.reset()
            batched.reset()
        dist.barrier(device_ids=[rank])
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29683"
    mp.spawn(check_rank, nprocs=4, join=True)
