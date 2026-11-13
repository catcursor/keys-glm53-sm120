# Keys GLM-5.3 EXL3 on four SM120 GPUs

This project runs the **full Keys GLM-5.3 EXL3 2.75 BPW checkpoint** on one Linux host with four **NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition** GPUs (SM120, 96 GB each, 300 W per card). TensorFold serves the OpenAI-compatible API on rank 0 and runs one CUDA rank per GPU. The measured DSpark preset uses tensor parallelism (TP4), decode context parallelism (DCP4), and a **685,000-token FP8 KV pool shared by up to four active requests**.

## Upstream sources

| Source | How it is used |
| --- | --- |
| [TensorFold by Ash Hart and contributors](https://github.com/ashhart/TensorFold) and [drowzeys' CUDA/GLM fork at `00c3bb1`](https://github.com/drowzeys/TensorFold/commit/00c3bb11dc8b72fef0b26f2beb7a3e7f8c3a43ab) | Native Zig server, CUDA GLM-5.3 engine, EXL3 loading, sampling, and DCP. [Our patch](patches/tensorfold-sm120.patch) applies to this exact fork commit. |
| [TensorFold Python reference at `b6e3e8d`](https://github.com/drowzeys/TensorFold/commit/b6e3e8d16453a008f10450cd40b124d5b799b802) | Triton kernel source used during AOT compilation. The native server uses the compiled CUDA kernels during inference. |
| [drowzeys' four-DGX-Spark deployment](https://github.com/drowzeys/keys-GLM-5.3-EXL3-2.75BPW-MixedK-Ablit-MLX-Serve-4-DGXSparks) and its [TensorFold TP4 base recipe](https://github.com/drowzeys/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark) | Reference for the four-rank setup and launch/tile metadata. The asset script fetches the metadata from the pinned Spark base image; it builds fresh SM120 cubins. |
| [Z.ai GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), [drowzeys' gated Keys checkpoint](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated), and [Turboderp's ExLlamaV3](https://github.com/turboderp-org/exllamav3) | Original model, the pinned mixed-bit checkpoint used in this port, and the EXL3 format, respectively. The test used the checkpoint's 6-bit output head. |
| [NVIDIA CUDA and NCCL](https://developer.nvidia.com/nccl), [PyTorch](https://pytorch.org/), [Triton](https://github.com/triton-lang/triton), and [voipmonitor's CUDA image](https://hub.docker.com/r/voipmonitor/vllm) | Build/runtime environment. The local test image was derived from the pinned public image below. The service itself runs TensorFold's native binary. |
| [DSpark ft2 at `681d43d`](https://huggingface.co/drowzeys/keys-GLM-5.3-speculator.dspark-ft2/tree/681d43db532b2f004d21c1d5ebe4aef5678d564f) | Speculative drafter for the shared FP8 KV pool. Each request has its own drafter state. |
| [LMCache](https://github.com/LMCache/LMCache) | Used in the separate `dcp4-fp8-lmcache` preset. The shared presets use GPU prefix reuse. |
| [Local Inference Lab PCIe bandwidth and P2P measurements](https://github.com/local-inference-lab/rtx6kpro/blob/master/hardware/pcie-bandwidth.md) | Hardware reference for PCIe peer-to-peer bandwidth, latency, BAR1, and the NVIDIA ForceP2P override. |

The public repository contains source, a patch, and synthetic test results. Model weights, container layers, and compiled GPU kernels are obtained separately under their respective licenses. See [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attributions.

## How it runs

- The launcher starts four ranks in one container and assigns CUDA devices 0–3. NCCL handles intra-host PCIe communication. Rank 0 listens on `127.0.0.1:18992` and serves model ID `model`.
- DCP4 shards the KV cache across the four GPUs. Each MLA row has 512 E4M3 values, four FP32 scales, and 64 BF16 RoPE values. Index keys remain BF16.
- A request reserves its tokenized input plus its requested maximum output. The reservation is released when the request finishes. Up to four requests run when their reservations fit; a request waits if the remaining token budget or contiguous range is too small. A single request can reserve the whole pool.
- Decode uses CUDA Graphs. GPU prefix checkpoints can be reused across stream slots. Reusing a region evicts overlapping idle checkpoints.
- The DSpark preset verifies drafts from up to four requests in one target-model batch. With four streams, the verification window can hold up to 32 rows and each stream can propose up to seven draft tokens per round.

## Reproduce on Linux

Use an x86-64 host with four idle SM120 GPUs, a driver supporting CUDA 13.2, Docker with NVIDIA Container Toolkit, and enough local storage for the full gated checkpoint and build assets. The scripts expect Bash, Git, and Python 3.11+.

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
python3 -B prepare-assets.py drafter

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
    bash /port/compile-aot.sh dcp4-fp8-shared-dspark
  '

python3 -B launch_sm120.py start --preset dcp4-fp8-shared-dspark
curl --fail http://127.0.0.1:18992/v1/models
```

The kernel sweep must report zero missing variants. The launcher waits for all four ranks to capture graphs before reporting ready. For the HTTP smoke test, use an otherwise idle server:

```bash
python3 -B check_fp8_four_streams.py \
  --rank-log rank-logs-dcp4-fp8-shared-dspark/rank-0.log \
  --no-round-profile --shared-context 685000
```

## Latest DSpark performance

Measured on November 13, 2026 with the shared FP8 KV and DSpark ft2 preset. All four Max-Q GPUs reported PCIe Gen5 ×16 links and a 300 W power limit. Resizable BAR and ForceP2P were enabled, BAR1 was 128 GiB per GPU, and the driver reported peer reads and writes available between every pair of GPUs. The prefill exchange passed the BF16 startup checks against NCCL 2.30.4. Each rank captured 663 CUDA Graphs.

### Cold prefill

Two runs per prompt length. All six retrieval responses were correct, with zero cached input tokens and zero eager decode steps.

| Prompt | Cold prefill (tok/s) |
| --- | ---: |
| 16K | 3,600.7–3,630.8 |
| 32K | 3,592.9–3,665.5 |
| 66K | 3,445.4–3,458.2 |

### Long-form prose decode

Each request asked for a long English essay on a different topic. Prompts were 110–113 tokens, with temperature 0 and thinking disabled. Every request produced 2,048 visible output tokens. Aggregate decode throughput divides the batch's output tokens by the time from the first streamed text to the last streamed text. The four-request batch briefly queued during admission, then all four streams produced output concurrently for 33.6 seconds. The measured batches had zero eager decode steps.

| Concurrent requests | Total output tokens | Decode interval (s) | Aggregate decode (tok/s) | Per-request decode (tok/s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2,048 | 19.23 | 106.5 | 106.5 |
| 4 | 8,192 | 50.56 | 162.0 | 41.0–60.2 |

In separate correctness tests, the same build passed short and 15K SQL checks, including a mixed workload. Four overlapping 10K retrieval requests returned their expected answers, and a warm request reused a 16,384-token prefix.
