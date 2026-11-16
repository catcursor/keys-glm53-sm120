# Keys GLM-5.3 EXL3 on four SM120 GPUs

This project runs the **full Keys GLM-5.3 EXL3 2.75 BPW checkpoint** on one Linux host with four **NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition** GPUs (SM120, 96 GB each; latest decode measurements at 268 W per card). TensorFold serves the OpenAI-compatible API on rank 0 and runs one CUDA rank per GPU. The measured DSpark preset uses tensor parallelism (TP4), decode context parallelism (DCP4), and a **1,000,000-token (1M) FP8 KV pool shared by up to four active requests**, with **CUDA Graphs disabled**.

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
- The current measured native TensorFold deployment disables CUDA Graphs (`TF_GLM53_GRAPHS=0`). GPU prefix checkpoints can be reused across stream slots. Reusing a region evicts overlapping idle checkpoints.
- The Owner Exchange preset saves intermediate prefix checkpoints every 8,192 tokens, matching its 8,192-row prefill chunks. Set `TF_GLM53_PREFILL_PROFILE=1` when starting the launcher to log rank 0's prefill phase timings, including staging the DCP history.
- The Owner Exchange preset enables `TF_GLM53_PREFILL_AG_IPC=1` to gather prefill rows through CUDA IPC peer copies. Startup compares all three gather kinds against NCCL byte for byte across several row counts and both buffer halves. A failed equality check disables peer gathers on all ranks.
- The DSpark preset verifies drafts from up to four requests in one target-model batch. With four streams, the verification window can hold up to 32 rows and each stream can propose up to seven draft tokens per round.

## Decode attention without changing precision

The FP8 DCP decode kernel avoids two sources of redundant work while keeping the cache format and arithmetic precision unchanged:

- A fully masked tile skips its unused query/key dot products and softmax. It still executes the original-shaped value dot and accumulator updates on opaque zero/one operands, preserving the empty-tile floating-point behavior.
- Each cache row's four FP32 scales are loaded once as four values and broadcast over their 128-value groups. Every latent element still undergoes the same FP8-to-FP32 conversion, one FP32 multiplication and BF16 conversion.

These paths apply to the existing `CHK=256, KTT=32` decode tiling; the scale path additionally requires `LW=512`. Query/key and value dot products for valid tiles, softmax reductions, chunk/rank ordering and sampling are unchanged. KV writers, unpack and prefill kernels retain their existing implementations. No new lower-precision computation or approximate attention is introduced.

On SM120, the tested attention implementation passed 24 direct loader checks and a 986-check matrix with bitwise-identical outputs, including special floating-point values and mixed stream slots. The end-to-end measurements and additional correctness checks are recorded below. The committed attention functions and their transitive JIT dependencies match that tested implementation; the unused experimental FMA helper is excluded.

The native runtime consumes AOT cubins, so editing this Python file alone does not update a running server. Rebuild the decode AOT entries with the existing FP8 launch settings (`num_warps=4, num_stages=1`) and run variant coverage and model validation. When updating the audited 370-entry production pack, replace all 36 matching decode records with their 24 unique newly compiled keys, retain the other 334 records and cubins byte for byte, and require one match per target launch. The result has 358 records. Appending a second set with a merge-only operation can leave duplicate matching keys and select an older kernel. Fresh builds can use the repository's normal `compile-aot.sh` workflow; the 334-record retention rule applies specifically to replacing the audited existing pack.

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
    bash /port/compile-aot.sh dcp4-fp8-shared-dspark
  '

python3 -B launch_sm120.py start --preset dcp4-fp8-shared-dspark
curl --fail http://127.0.0.1:18992/v1/models
```

The kernel sweep must report zero missing variants. The existing launcher recipe above uses CUDA Graphs and its 685,000-token preset; the current 1M graph-off deployment is documented below. The launcher waits for all four ranks to capture graphs before reporting ready. For the HTTP smoke test, use an otherwise idle server:

```bash
python3 -B check_fp8_four_streams.py \
  --rank-log rank-logs-dcp4-fp8-shared-dspark/rank-0.log \
  --no-round-profile --shared-context 685000
```

## Current decode performance: 268 W, CUDA Graphs off, 1M context

Measured on November 17, 2026 (Asia/Shanghai), with TP4/DCP4, shared FP8 KV, DSpark ft2 and `TF_GLM53_GRAPHS=0`. Both baseline and optimized runs used **268 W per GPU** and identical CPU/NUMA settings. The original native executable, weights, quantization, sampling policy and 1,000,000-token shared KV allocation were retained. Only the decode attention AOT variants changed.

Each case used one warmup followed by five measured requests with a warm prefix cache, `temperature=0`, seed `20261115` and 512 generated tokens. Values below are median server-reported decode throughput; they exclude prefill. The [measurement record](benchmark-decode-268w.json) includes every measured throughput sample and artifact hashes.

| Workload | Baseline tok/s | Optimized tok/s | Improvement |
| --- | ---: | ---: | ---: |
| Short prose | 72.1 | 81.6 | **13.18%** |
| Code | 104.9 | 116.3 | **10.87%** |
| Prose with a 16K prompt | 64.8 | 70.4 | **8.64%** |

All three cases exceed the 3% retention threshold. Prompt/token hashes, generated output, verification rounds and draft acceptance counts matched the baseline. Cold/warm/changed-tail 32K retrieval checks and four-stream checks passed. The combined kernel also passed 24 direct-loader and 986 attention bitwise checks; the preceding tail change independently passed 986 checks. All 24 sampled draft-on/off pairs passed for short and 16K inputs, two sampling configurations and solo/four-stream execution, and matched the prior bitwise-validated candidate across runs.

All four ranks captured zero CUDA Graphs. Startup and the model API confirm a **1,000,000-token shared pool** (13.22 GiB per GPU); this is a total pool for up to four active requests. These tests do not measure full-1M-prompt throughput or retrieval accuracy. The 300 W prefill and decode results below are earlier measurements with their original power setting.

## Previous DSpark performance: 300 W, CUDA Graphs off, 1M context

Measured on November 15, 2026 on the running production TensorFold build, using four RTX PRO 6000 Blackwell Max-Q GPUs with **300 W power limits per GPU**, TP4/DCP4, shared FP8 KV, DSpark ft2, and `TF_GLM53_GRAPHS=0`. CPU/NUMA settings remained unchanged throughout this run. Startup captured zero CUDA Graphs on all four ranks, and every measured request logged zero graph replays. These are **direct measurements**, not power-normalized estimates.

The runtime includes the peer-gather change from TensorFold `b3c5c27` (production source snapshot `dcb39b9`), an 8,192-token prefill chunk and checkpoint stride, and `TF_GLM53_PREFILL_AG_IPC=1`. The archived failed decode experiment is not part of this measured build.

### Code and prose decode

Two batches per workload and concurrency. The requests reuse the original long-form literary-story and Python-project workloads: 92–117 prompt tokens, temperature 0, thinking disabled, and **2,048 visible output tokens per request**. Each four-request batch produces 8,192 tokens. Aggregate throughput is total completion tokens divided by the interval from the earliest first streamed content to the latest last streamed content; prefill is excluded. The displayed result is the median of two batches.

| Workload | Concurrent requests | Current runs (tok/s) | Current median (tok/s) | Earlier graph-on result (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| Literary prose | 1 | 71.6–72.0 | 71.8 | 65.7 |
| Literary prose | 4 | 129.5–129.6 | 129.6 | 115.6 |
| Python code | 1 | 108.5–108.5 | 108.5 | 96.7 |
| Python code | 4 | 173.2–173.4 | 173.3 | 153.1 |

**With CUDA Graphs disabled, no decode throughput regression was observed against the earlier published graph-on results for these workloads.** The earlier values come from the [original DSpark benchmark](https://github.com/catcursor/keys-glm53-sm120/blob/cb9cf84008ad4990ed09cafac09289a3835e1280/README.md#dspark-validation-and-decode-performance). This is a historical comparison across runtime revisions and shared-pool sizes, not an isolated measurement of CUDA Graph overhead. The literary-story prompts here differ from the long-essay prompts used in the later 106.5/162.0 tok/s table; those figures are not directly comparable.

### Cold prefill

Two runs at each of the same three context-length tiers used in the previous README. Each prompt contains a verification code near the beginning followed by repeated inventory text. A different leading run marker prevents prefix reuse. All six requests had **zero cached tokens** and returned the exact expected code. Prefill throughput is actual prompt tokens divided by the server's logged prefill time; it excludes decoding the 256-token answer.

| Prompt tier | Actual prompt tokens | Current cold prefill (tok/s) | Earlier graph-on prefill (tok/s) | Correct retrievals |
| --- | ---: | ---: | ---: | ---: |
| 16K | 16,079 | 3,774.4–3,883.8 | 3,600.7–3,630.8 | 2/2 |
| 32K | 32,079 | 3,864.9–3,945.8 | 3,592.9–3,665.5 | 2/2 |
| 66K | 66,078 | 3,868.7–3,889.2 | 3,445.4–3,458.2 | 2/2 |

These prefill results also exceed the [previously published graph-on ranges](https://github.com/catcursor/keys-glm53-sm120/blob/86381ca346f80345f4dd178c7a947143ce80fc75/README.md#cold-prefill). As with decode, this compares complete deployments rather than isolating the graph setting.

### 1M shared context

The running graph-off deployment has increased its **configured maximum context to 1,000,000 tokens**. Startup allocated the shared pool successfully, and both `/v1/models` and `/tokenize` report `max_model_len=1000000`. The startup log reports **13.22 GiB of shared KV per GPU**; approximately **4.2 GiB per GPU remained free** during these measurements.

This is one total pool shared by up to four active requests. A request's input plus reserved output must fit within it; it is not 1M tokens per concurrent request. The 1M figure is the verified running configuration and allocation. This benchmark measures prefill only at the three lengths above and does not establish full-1M-prompt throughput or retrieval accuracy.

The measured runtime uses the existing model, AOT pack, output-head and DSpark settings, with these capacity settings inside its configured container:

```bash
export TF_GLM53_GRAPHS=0
export TF_GLM53_SHARED_KV=1
export TF_GLM53_FP8_KV=1
export TF_GLM53_DCP=4
export TF_GLM53_DCP_STREAMS_RESERVE_GB=12
bash /port/run-ranks.sh 18992 1000000 4
```

The existing `launch_sm120.py` and `check_fp8_four_streams.py` graph-capture checks still describe the graph-on validation recipe above; they are not the graph-off benchmark runner.

## DSpark prefix-reuse regression

Long prompts can save intermediate prefix checkpoints for a shared system prompt or a conversation branch. The prefill fast path used to capture draft activations only near the end of the full prompt. Saving an earlier checkpoint could therefore copy unwritten or stale rows from the DSpark KV ring. The fix makes activation capture and ring writes use the same cutoff for each chunk, covering the next saved checkpoint and the final prompt while still skipping rows neither needs.

The regression test on the four-GPU Max-Q host sent 211,007 input tokens containing synthetic malformed text, cancelled after generation began, and retried the exact prompt. The retry restored all 211,007 tokens from cache, recovered the expected verification code, and generated 128 tokens. A new chat sharing 8,233 system-prefix tokens and an independent new chat also returned their expected codes without a repeated punctuation loop. The service used the 685,000-token shared FP8 KV pool with four streams configured. Each rank captured 663 CUDA Graphs, and the requests logged zero eager steps.

## DCP4 long-context index writes

The merged AOT pack on the four-GPU host contained two `_ik_write` variants per rank with the same DCP4 row-mode launch signature. The old variant wrote to `BASE + POS` on every rank. The patched variant checks the owning rank and writes to `BASE + POS / 4`. TensorFold could select the old cubin by hash order, causing out-of-bounds writes and corrupt output around 171K tokens in the 685K shared pool.

The DCP4 build patches the Triton writers before compiling and validates the merged catalog. The launcher also checks existing packs before starting. To repair an affected pack in place, run `python3 -B aot_index_writers.py PATH/TO/pack/aot.json --repair`. The repair removes the four known stale entries only when compatible replacements exist. If it reports a missing replacement or another conflict, rebuild the pack.
