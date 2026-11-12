# Keys GLM-5.3 EXL3 on four SM120 GPUs

This project runs the **full Keys GLM-5.3 EXL3 2.75 BPW checkpoint** on one Linux host with four NVIDIA RTX PRO 6000 Blackwell GPUs. TensorFold serves the OpenAI-compatible API on rank 0 and runs one CUDA rank per GPU. The tested configuration combines tensor parallelism (TP4), decode context parallelism (DCP4), and a **728,888-token FP8 KV pool shared by up to four active requests**.

## Upstream sources

| Source | How it is used |
| --- | --- |
| [TensorFold by Ash Hart and contributors](https://github.com/ashhart/TensorFold) and [drowzeys' CUDA/GLM fork at `00c3bb1`](https://github.com/drowzeys/TensorFold/commit/00c3bb11dc8b72fef0b26f2beb7a3e7f8c3a43ab) | Native Zig server, CUDA GLM-5.3 engine, EXL3 loading, sampling, and DCP. [Our patch](patches/tensorfold-sm120.patch) applies to this exact fork commit. |
| [TensorFold Python reference at `b6e3e8d`](https://github.com/drowzeys/TensorFold/commit/b6e3e8d16453a008f10450cd40b124d5b799b802) | Triton kernel source used during AOT compilation. The native server uses the compiled CUDA kernels during inference. |
| [drowzeys' four-DGX-Spark deployment](https://github.com/drowzeys/keys-GLM-5.3-EXL3-2.75BPW-MixedK-Ablit-MLX-Serve-4-DGXSparks) and its [TensorFold TP4 base recipe](https://github.com/drowzeys/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark) | Reference for the four-rank setup and launch/tile metadata. The asset script fetches the metadata from the pinned Spark base image; it builds fresh SM120 cubins. |
| [Z.ai GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), [drowzeys' gated Keys checkpoint](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated), and [Turboderp's ExLlamaV3](https://github.com/turboderp-org/exllamav3) | Original model, the pinned mixed-bit checkpoint used in this port, and the EXL3 format, respectively. The test used the checkpoint's 6-bit output head. |
| [NVIDIA CUDA and NCCL](https://developer.nvidia.com/nccl), [PyTorch](https://pytorch.org/), [Triton](https://github.com/triton-lang/triton), and [voipmonitor's CUDA image](https://hub.docker.com/r/voipmonitor/vllm) | Build/runtime environment. The local test image was derived from the pinned public image below. The service itself runs TensorFold's native binary. |
| [DSpark ft2](https://huggingface.co/drowzeys/keys-GLM-5.3-speculator.dspark-ft2) and [LMCache](https://github.com/LMCache/LMCache) | Used in separate experiments. The shared-pool run below used GPU prefix reuse and had speculative decoding and LMCache disabled. |

The public repository contains source, a patch, and synthetic test results. Model weights, container layers, and compiled GPU kernels are obtained separately under their respective licenses. See [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attributions.

## How it runs

- The launcher starts four ranks in one container and assigns CUDA devices 0–3. NCCL handles intra-host PCIe communication. Rank 0 listens on `127.0.0.1:18992` and serves model ID `model`.
- DCP4 shards the KV cache across the four GPUs. Each MLA row has 512 E4M3 values, four FP32 scales, and 64 BF16 RoPE values. Index keys remain BF16.
- A request reserves its tokenized input plus its requested maximum output. The reservation is released when the request finishes. Up to four requests run when their reservations fit; a request waits if the remaining token budget or contiguous range is too small. A single request can reserve the whole pool.
- Decode uses CUDA Graphs. GPU prefix checkpoints can be reused across stream slots. Reusing a region evicts overlapping idle checkpoints.

## Reproduce on Linux

Use an x86-64 host with four idle SM120 GPUs, a driver supporting CUDA 13.2, Docker with NVIDIA Container Toolkit, and enough local storage for the full gated checkpoint and build assets. The measured GPUs were RTX PRO 6000 Blackwell Max-Q Workstation Edition cards with 96 GB each. The scripts expect Bash, Git, and Python 3.11+.

Accept the model's terms on Hugging Face and authenticate with the Hugging Face CLI before downloading the pinned checkpoint. From the root of this repository:

```bash
export TF_SM120_MODEL="$PWD/models/keys-glm53"
export TF_SM120_IMAGE='docker.io/voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b'

hf download drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated \
  --revision dac482e56c6346a12fff768a07bbb6bac76d7ef4 \
  --local-dir "$TF_SM120_MODEL"

bash bootstrap-tensorfold.sh
python3 -B prepare-assets.py reference
python3 -B prepare-assets.py templates

curl -fLO https://ziglang.org/download/0.17.0/zig-x86_64-linux-0.17.0.tar.xz
printf '%s  %s\n' \
  1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026 \
  zig-x86_64-linux-0.17.0.tar.xz | sha256sum -c
mkdir -p zig
tar -xJf zig-x86_64-linux-0.17.0.tar.xz -C zig --strip-components=1

docker run --rm --gpus '"device=0"' --ipc=host \
  -v "$PWD:/port" -v "$TF_SM120_MODEL:/model:ro" \
  -e TF_SM120_PREFIX=/port/build-dcp4-fp8-shared \
  --entrypoint /bin/bash "$TF_SM120_IMAGE" -ec '
    bash /port/build-sm120.sh /port
    python3 -B /port/TensorFold/tools/glm53/patch_fused_dcp_rows.py \
      /port/python-reference/src/tensorfold/families/glm_moe_dsa/cuda/fused.py
    bash /port/compile-aot.sh dcp4-fp8-shared
  '

python3 -B launch_sm120.py start --preset dcp4-fp8-shared
curl --fail http://127.0.0.1:18992/v1/models
```

The kernel sweep must report zero missing variants. The launcher waits for all four ranks to capture graphs before reporting ready. For the HTTP smoke test, use an otherwise idle server:

```bash
python3 -B check_fp8_four_streams.py \
  --rank-log rank-logs-dcp4-fp8-shared/rank-0.log \
  --no-round-profile --shared-context 728888
```

## Measured run

| Check | Result |
| --- | --- |
| GPU topology and software | 4× RTX PRO 6000 Blackwell Max-Q, SM120, one x86-64 Linux host; NVIDIA driver 615.71.09, CUDA 13.2.1, NCCL 2.30.4, PyTorch 2.12.0+cu132, Triton 3.7.0, Zig 0.17.0 |
| Shared capacity | 728,888 tokens exposed by the API, 9.60 GiB KV allocation per GPU |
| GPU memory after startup | 85,194 MiB used and 12,148 MiB free per GPU |
| CUDA Graphs | 80 captured per rank; zero eager decode steps in the four-request test |
| Concurrent HTTP requests | Four active at the same time; all four returned their own expected code |
| GPU prefix reuse | 2,583 tokens restored across slots |
| Long prompt and queueing | 195,032 real input tokens, 116.30 s prefill. The request reserved all 728,888 tokens, returned the expected code, and the waiting request completed after the reservation was released. |
| Build checks | 115/115 tests passed; AOT coverage 254/254 for the shared shape |

The 728,888-token figure is the configured capacity and was exercised as a reservation. The longest actual prompt in this run was **195,032 tokens**. The run does not establish full-window generation speed or accuracy. See [the sanitized measurement record](results/shared-pool.json).
