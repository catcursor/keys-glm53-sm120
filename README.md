# Keys GLM-5.3 EXL3 on four SM120 GPUs

The full **Keys GLM-5.3 EXL3 2.75 BPW checkpoint** runs through TensorFold's native OpenAI-compatible server with TP4/DCP4, DSpark ft2, **CUDA Graphs disabled**, and a **1,000,000-token FP8 KV pool shared by up to four requests**.

## Hardware and current configuration

Configuration measured on November 17, 2026 (Asia/Shanghai). Deployment settings and persistence are documented in [DEPLOYMENT-180.md](DEPLOYMENT-180.md).

| Component | Configuration |
| --- | --- |
| GPUs | 4 × NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 96 GB each; **300 W per GPU** |
| CPUs | 2 × Intel Xeon Gold 6454S; 64 physical cores total, SMT disabled; Turbo enabled, up to 3.4 GHz |
| CPU power limits | PL1 **170 W**, PL2 **220 W**, per socket |
| Host memory | **94 GiB** visible to Linux |
| Inference placement | NUMA node 0 CPU and memory; rank CPU sets `0-7`, `8-15`, `16-23`, `24-31` |
| Native snapshot staging | **1 GiB pinned host memory per rank**, 4 GiB total; one pending snapshot per rank |
| LMCache L1 | **32 GiB RAM**; CPUs `32-39`, NUMA memory node 1 |
| LMCache L2 | **256 GiB disk**, under `/u2/cache/keys-glm53-lmcache-v2` |
| ZFS ARC | **2 GiB maximum**, persisted across reboots |
| NVIDIA driver | **615.71.09** |

## Current performance

Samples, timings and runtime identities are in [benchmark-300w.json](benchmark-300w.json); [benchmark_current.py](benchmark_current.py) contains the exact workload and timing procedure. All results below use the asynchronous RAM/disk cache, CUDA Graphs off, and the hardware configuration above.

### Decode

Each workload/concurrency combination runs twice, with `temperature=0`, thinking disabled and **2,048 visible output tokens per request**. Throughput is total visible tokens divided by the client-observed interval from the first streamed content to the last streamed content across the batch. The table reports the median of the two batches. This interval includes any cache-related pauses that occur within it; it is not a sum of server-reported per-request speeds.

| Workload | Concurrent requests | Runs (tok/s) | Median aggregate (tok/s) |
| --- | ---: | ---: | ---: |
| Literary prose | 1 | 80.7–80.8 | **80.7** |
| Literary prose | 4 | 136.0–136.9 | **136.4** |
| Python code | 1 | 117.8–118.2 | **118.0** |
| Python code | 4 | 179.4–179.4 | **179.4** |

### Cold prefill

Each tier runs twice with a distinct leading marker, **zero cached tokens**, and a verification code near the beginning of the prompt. Prefill speed is actual prompt tokens divided by the server's logged prefill duration. Both runs return the exact expected code. Total request time includes prefill, 256 output tokens and response completion; background cache persistence can continue afterward.

| Prompt tier | Actual prompt tokens | Cold prefill (tok/s) | Total request (s) | Correct |
| --- | ---: | ---: | ---: | ---: |
| 16K | 16,086 | 3,655.9–3,723.6 | 8.30–8.46 | 2/2 |
| 32K | 32,086 | 3,838.0–3,842.6 | 12.16–12.25 | 2/2 |
| 66K | 66,085 | 3,853.4–3,855.6 | 21.02–21.09 | 2/2 |

The **1M context is a verified configuration and allocation**, not a full-1M-prompt performance or retrieval result. The pool is shared across requests; each request reserves its input plus its maximum output, and waits when the remaining pool cannot accommodate it.

## Serving and cache behavior

- Four CUDA ranks serve model ID `model`; rank 0 exposes the API at `http://127.0.0.1:18992`.
- Responses finish before optional background cache uploads. Each rank has one **1 GiB pinned snapshot buffer**; busy or oversized snapshots are skipped. L2 writes use a bounded **512 MiB queue**, with backpressure confined to background workers.
- GPU prefix reuse remains available during generation; external restoration and index publication run when no stream is active. Missing or damaged chunks fall back to cold prefill. Graceful shutdown drains L2 writes; crashes can lose pending writes.
- The **1M inference context remains available**, while the snapshot cap limits which prefixes can be stored externally. GPU copies still consume bandwidth.
- [Validation](async-cache-validation.json) passed 22 backend tests, native host tests, four-stream slot reuse and 66K RAM/disk restoration. All four new requests produced correct output while the completed 66K request was still uploading. The decode rerun matched all **20 prior outputs and token counts** (40,960 visible tokens); raw generated token IDs are not exposed by the API.

See [DEPLOYMENT-180.md](DEPLOYMENT-180.md) for the native compatibility patch, NUMA settings, cache budgets and runtime locations. The current **1M / four-stream / L1+L2 deployment requires that native integration**; the generic launcher command below does not reproduce it by itself.

## Reproduction and build assets

Use Linux x86-64, four SM120 GPUs, a CUDA 13.2-compatible driver, Docker with NVIDIA Container Toolkit, Bash, Git and Python 3.11+. Accept the gated checkpoint's terms and authenticate with Hugging Face before downloading it. Weights, container layers and compiled GPU kernels are distributed separately under their respective licenses.

<details>
<summary>Base build recipe — the existing 685K, graph-on launcher preset</summary>

This builds the repository's base artifacts. Its launcher and readiness checks require captured CUDA Graphs; apply the deployment-specific native/cache integration and settings in [DEPLOYMENT-180.md](DEPLOYMENT-180.md) for the configuration benchmarked above.

```bash
export TF_SM120_MODEL="$PWD/models/keys-glm53"
export TF_SM120_IMAGE='docker.io/voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b'
hf download drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated \
  --revision dac482e56c6346a12fff768a07bbb6bac76d7ef4 --local-dir "$TF_SM120_MODEL"
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
    bash /port/compile-aot.sh dcp4-fp8-shared-dspark
  '
python3 -B launch_sm120.py start --preset dcp4-fp8-shared-dspark
curl --fail http://127.0.0.1:18992/v1/models
python3 -B check_fp8_four_streams.py \
  --rank-log rank-logs-dcp4-fp8-shared-dspark/rank-0.log \
  --no-round-profile --shared-context 685000
```

The kernel sweep must report zero missing variants. The readiness check verifies graph capture on all four ranks for this base preset.

</details>

<details>
<summary>Kernel correctness and AOT maintenance</summary>

The decode attention implementation retains the FP8 cache format and existing arithmetic precision. It skips unused query/key and softmax work for fully masked tiles while retaining the original-shaped value accumulation, and loads each row's four FP32 scales once before broadcasting them over their 128-value groups. Valid-tile arithmetic, reductions, KV writers and prefill remain unchanged. The paths passed 24 loader checks and a 986-case bitwise attention matrix.

Inference consumes AOT cubins: editing [dcp4_fp8_kv.py](dcp4_fp8_kv.py) alone does not update a running server. Recompile with the existing decode launch settings (`num_warps=4, num_stages=1`), validate launch-key uniqueness and variant coverage, then run model correctness checks. The deployed pack contains 358 entries. When replacing entries in an existing pack, replace every matching decode launch record rather than appending duplicate keys.

[aot_index_writers.py](aot_index_writers.py) checks DCP4 index-writer variants. An affected catalog can be repaired with `python3 -B aot_index_writers.py PATH/TO/pack/aot.json --repair` only when compatible replacement kernels are already present; otherwise rebuild it. DSpark intermediate checkpoints must capture the activation and ring state for each saved prefill boundary.

</details>

## Upstream sources and attribution

| Source | Role |
| --- | --- |
| [TensorFold — Ash Hart and contributors](https://github.com/ashhart/TensorFold), [drowzeys' CUDA/GLM fork at `00c3bb1`](https://github.com/drowzeys/TensorFold/commit/00c3bb11dc8b72fef0b26f2beb7a3e7f8c3a43ab) | Native server, EXL3 loading, sampling and DCP; [base patch](patches/tensorfold-sm120.patch) targets this fork revision. |
| [TensorFold Python reference at `b6e3e8d`](https://github.com/drowzeys/TensorFold/commit/b6e3e8d16453a008f10450cd40b124d5b799b802) | Triton kernel source for SM120 AOT compilation. |
| [drowzeys' four-DGX-Spark deployment](https://github.com/drowzeys/keys-GLM-5.3-EXL3-2.75BPW-MixedK-Ablit-MLX-Serve-4-DGXSparks), [TP4 base recipe](https://github.com/drowzeys/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark) | Four-rank setup, launch metadata and tile metadata. |
| [Z.ai GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), [drowzeys' Keys checkpoint](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated), [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | Original model, mixed-bit checkpoint with its 6-bit output head, and EXL3 format. |
| [DSpark ft2 at `681d43d`](https://huggingface.co/drowzeys/keys-GLM-5.3-speculator.dspark-ft2/tree/681d43db532b2f004d21c1d5ebe4aef5678d564f), [LMCache](https://github.com/LMCache/LMCache) | Speculative drafter and the native state-cache transport. |
| [NVIDIA CUDA/NCCL](https://developer.nvidia.com/nccl), [PyTorch](https://pytorch.org/), [Triton](https://github.com/triton-lang/triton), [voipmonitor's CUDA image](https://hub.docker.com/r/voipmonitor/vllm) | Build and runtime dependencies; inference uses TensorFold's native executable. |
| [Local Inference Lab](https://github.com/local-inference-lab/rtx6kpro/blob/master/hardware/pcie-bandwidth.md) | PCIe bandwidth, latency and peer-access reference. |

See [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution and licensing.
