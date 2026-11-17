# 180 deployment configuration

This deployment keeps CUDA Graphs disabled, the accepted Scale4/Tail-PV AOT
pack, a shared 1,000,000-token GPU KV pool, four concurrent streams, and DSpark.
The hardware and cache settings below describe the deployment measured in README.

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
- Disk L2: 256 GiB of payload, bounded LRU and synchronous write-through,
  under `/u2/cache/keys-glm53-lmcache-v2`.
- The CPU-only cache service uses CPUs `32-39` and NUMA memory node 1, leaving
  node 0's memory available for model startup, GPU transfer buffers and ranks.
  Its container limit is 36 GiB including Python, index and transfer overhead;
  swap is disabled for this container.
- The server listens only on `127.0.0.1:18553`.
- Each rank uses a 64 GiB learned-state index budget. The four rank shards
  together share the server's single global L1/L2 budgets.
- Old learned-state indexes were deleted. Model files and compiled AOT
  artifacts were retained.

`lmcache_tensorfold_server.py` implements the existing native LMCache wire
protocol with independent RAM/disk budgets. Atomic payload files, a SQLite
index and SHA256 checks prevent incomplete/corrupt payloads from being returned
as valid cached data. Committed L2 data survives server restarts. An abrupt stop
may lose recent LRU ordering, but not an already committed payload.

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

Cache storage is synchronous and can add request-completion latency. This feature
targets repeated prompt prefill/TTFT, and is not a claim of faster per-token decode.
GPU-resident hot prefixes continue to use the native GPU prefix cache.

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

The tiered backend passed 11 focused tests in the actual deployment image,
including native wire messages, restart recovery, L1/L2 eviction, corruption,
failed overwrite and concurrent rank writes. End-to-end checks passed for cold
prefill, GPU reuse, RAM restore, disk restore, four concurrent RAM/disk restores,
and a deliberately missing chunk on one rank followed by cold fallback and repair.
Restored text and token counts matched the accepted native baseline. The API does
not expose raw generated token IDs, so this is not a claim of raw token-ID parity.

The deployed native SHA256 is
`f0d2d178c3c760b814d57c600f3a02c684e7dcd3a9e5e7b3c8e1ede23b1bbaeb`.
Client stream throughput includes any synchronous cache pauses; native per-request
decode throughput is recorded separately in the benchmark evidence.
