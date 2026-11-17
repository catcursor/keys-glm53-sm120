# 180 deployment configuration

This deployment keeps CUDA Graphs disabled, the accepted Scale4/Tail-PV AOT
pack, a shared 1,000,000-token GPU KV pool, four concurrent streams, and DSpark.
The hardware settings below match README. The asynchronous cache integration and
its passed integration checks are documented here.

## Host settings

- ZFS ARC maximum: 2 GiB; minimum: 512 MiB. The minimum must be below the new
  maximum. `/etc/modprobe.d/zfs-arc-limit.conf` is included in the updated
  initramfs. Existing ARC/page caches were cleared once after setting the cap.
- Intel Turbo enabled (`no_turbo=0`), allowing up to 3.4 GHz. The existing
  `powersave` governor and `balance_performance` EPP are retained.
- CPU package power limits remain PL1 170 W and PL2 220 W on both sockets.
- All four GPU power caps are 300 W for the current measurements. A systemd
  override also preserves 300 W at boot; the older unit filename still contains `258`.
- GPU ranks use NUMA node 0 for CPU and memory placement. Rank CPU sets are
  `0-7`, `8-15`, `16-23`, and `24-31`.
- NCCL and P2P settings are unchanged.

## Cache layout

- RAM L1: 32 GiB of payload, bounded LRU.
- Disk L2: 256 GiB of payload, bounded LRU with background write-back,
  under `/u2/cache/keys-glm53-lmcache-v2`.
- Native snapshot staging: one preallocated 1 GiB pinned host buffer per rank,
  4 GiB total, controlled by `TF_GLM53_LMCACHE_STAGE_MIB=1024`. Each rank allows
  one pending snapshot; a busy stage or an oversized complete state is skipped.
- L2 write queue: 512 MiB and at most 128 chunks, including the active write,
  configured with `--write-queue-mib 512`. Payload references are immutable and
  shared with L1. When full, the upload connection waits for capacity for at most
  1 second (`--write-queue-wait-ms 1000`), releasing the RAM lock. GET and other
  connections can continue. Only a timeout skips L2 while retaining the L1 copy;
  `--write-queue-wait-ms 0` selects immediate skipping for overload tests.
- The CPU-only cache service uses CPUs `32-39` and NUMA memory node 1, leaving
  node 0's memory available for model startup, GPU transfer buffers and ranks.
  Its container limit is 36 GiB including Python, index, queue and transfer overhead;
  swap is disabled for this container.
- The server listens only on `127.0.0.1:18553`.
- Each rank uses a 64 GiB learned-state index budget. The four rank shards
  together share the server's single global L1/L2 budgets.
- Old learned-state indexes were deleted. Model files and compiled AOT
  artifacts were retained.

`lmcache_tensorfold_server.py` implements the existing native LMCache wire
protocol with independent RAM/disk budgets. PUT inserts an immutable chunk into
L1 and queues an L2 write, so its acknowledgement promises availability, not
durability. Disk I/O runs outside the RAM lock. Atomic payload files, a SQLite
index and SHA256 checks prevent incomplete/corrupt payloads from being returned
as valid cached data. Committed L2 data survives server restarts. Graceful
cache-server shutdown drains the write queue; a forced exit can lose queued
writes and recent LRU ordering. Before a persistence check, wait for
`l2_pending_bytes=0`, `l2_waiting_puts=0` and `l2_retired_chunks=0`, then verify no
write/cleanup errors or backpressure timeouts. The wait applies to native
background upload workers, not the inference scheduler. Queue, L1 and container
memory budgets remain unchanged.

The native compatibility patch is
`patches/tensorfold-sm120-multi-lmcache-v2.patch`. It is relative to the production
TensorFold source recorded in the adjacent manifest, not an arbitrary upstream
checkout. It adds complete-prompt snapshots for shared-KV multi-stream serving,
including carry, head logits and the DSpark ring. Cache identity includes the
native binary, AOT configuration, drafter checkpoint and quantized head identity.

Ordinary missing or damaged chunks cause all ranks to fall back to cold prefill.
Protocol or slot-preparation failures stop the ranks together instead of allowing
their states to diverge. Strict prompt extensions reuse an external snapshot only
when the original prefill chunk boundaries are compatible.

Finished responses emit DONE before optional snapshot work. Device-to-host copies
are queued on the existing compute stream, ordering them before later slot reuse.
The worker waits for the ready event and uploads CPU bytes; it does not access
live slots or the learned index. Index publication runs only while the scheduler
is idle. Admission while another stream is active skips external restoration;
GPU-resident hot-prefix reuse continues to work.

Asynchronous work still consumes GPU copy bandwidth, CPU time and memory. The
1 GiB per-rank snapshot cap does not reduce the 1M inference context, but it does
mean that not every long prefix can be cached externally. Busy/oversized snapshots
and L2 backpressure timeouts are best-effort skips, not inference failures. This feature
targets repeated prefill/TTFT and makes no claim of zero overhead.

## Runtime locations

Deployment root:
`/home/nono/llm-lab/runtime/keys-glm53-decode-20261116/deployment-20261117`

The directory contains the host configuration backup, native source/build,
rank logs, learned-state indexes, cache statistics and validation evidence.
`keys-glm53-native.service` and `keys-glm53-lmcache.service` are enabled at boot
and supervise the existing containers. Their mount dependencies ensure the model
and L2 storage are available before startup.

The launcher accepts `TF_GLM53_RANK_CPUSETS` (four semicolon-separated CPU sets)
and `TF_GLM53_LMCACHE_LEARN_GIB` (per-rank index budget). Container cpuset memory
placement supplies the NUMA binding.

## Validation

The asynchronous tiered backend passed 22 focused tests in the actual deployment
image, covering native wire messages, restart recovery, eviction, corruption,
bounded queues including active writes, slow-disk RAM responsiveness, concurrent
overwrite/read/remove races, bounded backpressure and close-time draining. The
native `test-cuda-host`, `test-server-cpu` and `test-glm53` build targets also
passed; the fail-fast build exited zero with the deployed binary hash.

[cache-validation.json](cache-validation.json) is the preceding synchronous-cache
baseline: cold prefill, GPU reuse, RAM/disk restoration, four concurrent restores
and missing-chunk fallback passed with matching text and token counts. It does
not establish correctness of the new asynchronous native integration. The API
does not expose raw generated token IDs, so these checks do not claim token-ID parity.

The asynchronous integration passed the checks summarized in
[async-cache-validation.json](async-cache-validation.json). Its 66,084-token
fixture and 256-token golden output come from the existing accepted
`cold-66k-r1` measurement; the four short-request goldens are unchanged. All seven
responses across cold output, four concurrent requests, RAM restoration and disk
restoration matched the exact golden text and prompt/completion/total counts.

The complete decode rerun also matched the preceding run for all 20 requests
across 8 batches: identical request hashes, exact generated text, content hashes
and token counts, with 2,048 visible tokens per request and 40,960 total. Matching
uses workload, concurrency, repetition and lane. This is additional long-output
parity evidence and makes no token-ID parity or performance-improvement claim.

The 66K request's final content arrived before DONE, which reached the client
7.6235 seconds before all ranks finished uploading. Four new requests became active and produced
correct output during that upload. Each rank staged 959,510,368 bytes within its
1 GiB cap, with one pending snapshot and four observed busy skips. The L2 queue
reached its 512 MiB bound without overflow, backpressure timeouts or I/O errors.

GPU-prefix flushes forced restoration from RAM and, after restarting only the
backend, disk. Both restored all 66,084 prompt tokens with exact golden output;
each read 916 chunks through the expected tier. The disk check verified a new
backend instance with empty L1. Pending writes, waiting PUTs and retired chunks
were zero before the restart. The final audit confirmed unchanged GPU power
limits, CPU power limits, NUMA placement and runtime environment.

The deployed native SHA256 is
`137526c8838494d2a30ede67b918a324a8ee5ae02d693d5e55287077e4bed123`.
The README speed figures were rerun on this asynchronous build. Client stream
throughput includes pauses inside the measured first-to-last-content interval;
native per-request decode throughput is recorded separately in the evidence.
