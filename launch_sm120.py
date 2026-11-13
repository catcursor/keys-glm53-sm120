"""Launch the four TensorFold ranks on one SM120 host."""

import argparse
import json
import os
import re
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL = Path(os.environ.get("TF_SM120_MODEL", ROOT / "models/keys-glm53")).expanduser().resolve()
REVISION = "dac482e56c6346a12fff768a07bbb6bac76d7ef4"
IMAGE = os.environ.get(
    "TF_SM120_IMAGE",
    "docker.io/voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b",
)
PORT = 5033
MASTER_PORT = 29681
CONTAINER = "keys-glm53-sm120"
RANK_LOGS = ROOT / "rank-logs"


@dataclass(frozen=True)
class Preset:
    container: str
    rank_logs: Path
    aot: str
    context: int
    parallel: int
    dcp: int
    prompt_rows: int
    native: str = "build/native/bin/tensorfold-native"
    gather_prefill: bool = False
    decode_packed: bool = False
    decode_a2a: bool = False
    decode_short_kv: bool = False
    fp8_kv: bool = False
    lmcache_port: int | None = None
    dspark: bool = True
    dcp_streams_reserve_gib: int | None = None
    shared_kv: bool = False


PRESETS = {
    "dcp1": Preset(CONTAINER, RANK_LOGS, "aot-sm120/pack", 32768, 4, 1, 8192),
    "dcp4-256k": Preset(
        "keys-glm53-sm120-dcp4", ROOT / "rank-logs-dcp4",
        "aot-sm120-dcp4-256k/pack", 262144, 1, 4, 1024,
    ),
    "dcp4-fast-prefill": Preset(
        "keys-glm53-sm120-dcp4-replicated", ROOT / "rank-logs-dcp4-replicated",
        "aot-sm120-dcp4-prefill-8192/pack", 262144, 1, 4, 8192,
        "build-dcp-prefill-replicated/native/bin/tensorfold-native", gather_prefill=True, decode_packed=True,
    ),
    "dcp4-packed-a2a": Preset(
        "keys-glm53-sm120-dcp4-packed-a2a", ROOT / "rank-logs-dcp4-packed-a2a",
        "aot-sm120-dcp4-packed-a2a/pack", 262144, 1, 4, 8192,
        "build-dcp-prefill-a2a/native/bin/tensorfold-native", gather_prefill=True, decode_packed=True, decode_a2a=True,
    ),
    "dcp4-short-kv": Preset(
        "keys-glm53-sm120-dcp4-short-kv", ROOT / "rank-logs-dcp4-short-kv",
        "aot-sm120-dcp4-prefill-8192/pack", 262144, 1, 4, 8192,
        "build-dcp-prefill-short-kv/native/bin/tensorfold-native", gather_prefill=True,
        decode_packed=True, decode_short_kv=True,
    ),
    "dcp4-fp8-lmcache": Preset(
        "keys-glm53-sm120-dcp4-fp8-lmcache", ROOT / "rank-logs-dcp4-fp8-lmcache",
        "aot-sm120-dcp4-fp8/pack", 524288, 1, 4, 8192,
        "build-dcp4-fp8-lmcache/native/bin/tensorfold-native", gather_prefill=True,
        decode_packed=True, decode_short_kv=True, fp8_kv=True, lmcache_port=18553,
    ),
    "dcp4-fp8-4streams": Preset(
        "keys-glm53-sm120-dcp4-fp8-4streams", ROOT / "rank-logs-dcp4-fp8-4streams",
        "aot-sm120-dcp4-fp8-4streams/pack", 182222, 4, 4, 8192,
        "build-dcp4-fp8-4streams/native/bin/tensorfold-native", gather_prefill=True,
        decode_packed=True, fp8_kv=True, dspark=False, dcp_streams_reserve_gib=18,
    ),
    "dcp4-fp8-shared": Preset(
        "keys-glm53-sm120-dcp4-fp8-shared", ROOT / "rank-logs-dcp4-fp8-shared",
        "aot-sm120-dcp4-fp8-4streams/pack", 728888, 4, 4, 8192,
        "build-dcp4-fp8-shared/native/bin/tensorfold-native", gather_prefill=True,
        decode_packed=True, fp8_kv=True, dspark=False, dcp_streams_reserve_gib=18, shared_kv=True,
    ),
    "dcp4-fp8-shared-dspark": Preset(
        "keys-glm53-sm120-dcp4-fp8-shared-dspark", ROOT / "rank-logs-dcp4-fp8-shared-dspark",
        "aot-sm120-dcp4-fp8-shared-dspark/pack", 728888, 4, 4, 8192,
        "build-dcp4-fp8-shared/native/bin/tensorfold-native", gather_prefill=True,
        decode_packed=True, fp8_kv=True, dcp_streams_reserve_gib=18, shared_kv=True,
    ),
    "dcp4-fp8-shared-dspark-64k": Preset(
        "GLM5.3", ROOT / "rank-logs-dcp4-fp8-shared-dspark-64k",
        "aot-sm120-dcp4-fp8-shared-dspark/pack", 65536, 4, 4, 8192,
        "build-dcp4-fp8-shared/native/bin/tensorfold-native", gather_prefill=True,
        decode_packed=True, fp8_kv=True, dcp_streams_reserve_gib=18, shared_kv=True,
    ),
    "dcp4-fp8-shared-dspark-64k-a2a": Preset(
        "keys-glm53-sm120-dcp4-fp8-shared-dspark-64k-a2a", ROOT / "rank-logs-dcp4-fp8-shared-dspark-64k-a2a",
        "aot-sm120-dcp4-fp8-shared-dspark-a2a/pack", 65536, 4, 4, 8192,
        "build-dcp4-fp8-shared/native/bin/tensorfold-native", gather_prefill=True,
        decode_packed=True, decode_a2a=True, fp8_kv=True,
        dcp_streams_reserve_gib=18, shared_kv=True,
    ),
    "dcp4-fp8-shared-dspark-2m": Preset(
        "GLM5.3", ROOT / "rank-logs-dcp4-fp8-shared-dspark-64k",
        "aot-sm120-dcp4-fp8-shared-dspark/pack", 1048576, 4, 4, 8192,
        "build-dcp4-fp8-shared/native/bin/tensorfold-native", gather_prefill=True,
        decode_packed=True, fp8_kv=True, dcp_streams_reserve_gib=3, shared_kv=True,
    ),
    "dcp4-fp8-shared-dspark-512k": Preset(
        "GLM5.3", ROOT / "rank-logs-dcp4-fp8-shared-dspark-64k",
        "aot-sm120-dcp4-fp8-shared-dspark/pack", 524288, 4, 4, 8192,
        "build-dcp4-fp8-shared/native/bin/tensorfold-native", gather_prefill=True,
        decode_packed=True, fp8_kv=True, dcp_streams_reserve_gib=18, shared_kv=True,
    ),
    "dcp4-fp8-shared-dspark-256k": Preset(
        "keys-glm53-sm120-dcp4-fp8-shared-dspark-256k", ROOT / "rank-logs-dcp4-fp8-shared-dspark-256k",
        "aot-sm120-dcp4-fp8-shared-dspark/pack", 262144, 4, 4, 8192,
        "build-dcp4-fp8-shared/native/bin/tensorfold-native", gather_prefill=True,
        decode_packed=True, fp8_kv=True, dcp_streams_reserve_gib=18, shared_kv=True,
    ),
}


def run(*command, capture=False):
    result = subprocess.run(command, check=True, text=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT if capture else None)
    return result.stdout if capture else None


def preflight(preset=PRESETS["dcp1"]):
    marker = MODEL / ".download-complete"
    if marker.exists():
        state = json.loads(marker.read_text())
        expected = {"phase": "complete", "revision": REVISION, "verified_files": 93, "weight_shards": 80}
        if any(state.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Checkpoint verification marker does not match the pinned checkpoint")
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    required = shards | {"config.json", "tokenizer.json", "lm_head-6bit/lm_head-6bit.safetensors"}
    if len(shards) != 80 or any(not (MODEL / name).is_file() or not (MODEL / name).stat().st_size for name in required):
        raise RuntimeError("Missing target checkpoint shards")
    artifacts = [preset.native, f"{preset.aot}/aot.json"]
    if preset.dspark:
        artifacts.extend(("dspark-ft2/config.json", "dspark-ft2/model.safetensors"))
    for path in artifacts:
        if not (ROOT / path).is_file():
            raise RuntimeError(f"Missing port artifact: {path}")
    occupied = run("ss", "-Hln", f"( sport = :{PORT} or sport = :{MASTER_PORT} )", capture=True)
    if occupied.strip():
        raise RuntimeError(f"Port already in use: {occupied.strip()}")
    containers = set(run("docker", "ps", "-a", "--format", "{{.Names}}", capture=True).splitlines())
    if preset.container in containers:
        raise RuntimeError("A port test container already exists")
    _ids = os.environ.get("TF_SM120_GPUS", "")
    _args = ["--query-gpu=uuid,compute_cap,memory.used", "--format=csv,noheader,nounits"]
    if _ids:
        _args += ["-i", _ids]
    inventory = run("nvidia-smi", *_args, capture=True)
    devices = [tuple(field.strip() for field in row.split(",")) for row in inventory.splitlines()]
    if len(devices) != 4 or any(cap != "12.0" or int(used) > 256 for _, cap, used in devices):
        raise RuntimeError("Four idle SM120 GPUs required")
    return [uuid for uuid, _, _ in devices]


def start_lmcache(port):
    name = "keys-sm120-lmcache"
    current = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Status}}", name],
        text=True, capture_output=True,
    )
    if current.returncode != 0:
        run("docker", "run", "-d", "--name", name, "--init", "--network", "host",
            "-v", f"{ROOT}/lmcache_tensorfold_server.py:/app/lmcache_tensorfold_server.py:ro",
            "--entrypoint", "/opt/venv/bin/python", IMAGE, "-B", "/app/lmcache_tensorfold_server.py",
            "--port", str(port), "--gib", "16")
    elif current.stdout.strip() != "running":
        run("docker", "start", name)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise RuntimeError(f"LMCache did not bind its loopback port. Check docker logs {name}")


def start(devices, preset=PRESETS["dcp1"]):
    env = {
        "CUDA_VISIBLE_DEVICES": ",".join(devices),
        "TF_GLM53_WORLD": "4",
        "TF_GLM53_MASTER": f"127.0.0.1:{MASTER_PORT}",
        "TF_GLM53_ROCE": "0",
        "TF_GLM53_DCP": str(preset.dcp),
        "TF_GLM53_FP8_KV": str(int(preset.fp8_kv)),
        "TF_GLM53_DCP_PREFILL_GATHER": str(int(preset.gather_prefill)),
        "TF_GLM53_DCP_SHORT_KV": str(int(preset.decode_short_kv)),
        "TF_GLM53_DCP_DECODE_PACKED": str(int(preset.decode_packed)),
        "TF_GLM53_DCP_DECODE_A2A": str(int(preset.decode_a2a)),
        "TF_GLM53_NATIVE": f"/port/{preset.native}",
        "TF_GLM53_PROMPT_ROWS": str(preset.prompt_rows),
        "TF_GLM53_PROMPT_ROWS_SHORT": str(preset.prompt_rows if preset.gather_prefill else min(4096, preset.prompt_rows)),
        "TF_GLM53_K": "0",
        "TF_GLM53_GRAPHS": os.environ.get("TF_GLM53_GRAPHS", "1"),
        "TF_GLM53_PREWARM": "1",
        "TF_GLM53_PROMPT_REUSE": "1",
        "TF_GLM53_REUSE_STRIDE": "4096",
        "TF_GLM53_AOT": f"/port/{preset.aot}",
        "TF_GLM53_HEAD_Q": "/model/lm_head-6bit/lm_head-6bit.safetensors",
        "TF_GLM53_TIMEOUT": "1200",
        "NCCL_IB_DISABLE": "1",
        "NCCL_SOCKET_IFNAME": "=lo",
        "NCCL_MIN_NCHANNELS": "8",
        "NCCL_MAX_NCHANNELS": "8",
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,P2P",
    }
    if preset.parallel > 1 and preset.dcp > 1:
        env["TF_GLM53_DCP_STREAMS"] = "1"
    if preset.shared_kv:
        env["TF_GLM53_SHARED_KV"] = "1"
    if preset.dcp_streams_reserve_gib is not None:
        env["TF_GLM53_DCP_STREAMS_RESERVE_GB"] = str(preset.dcp_streams_reserve_gib)
    if os.environ.get("TF_GLM53_ROUND_PROFILE") == "1":
        env["TF_GLM53_ROUND_PROFILE"] = "1"
    if preset.dspark:
        env.update(TF_GLM53_DSPARK="/port/dspark-ft2", TF_GLM53_DSPARK_POLICY="confidence",
                   TF_GLM53_DSPARK_CONFIDENCE="0.3")
    if preset.lmcache_port:
        start_lmcache(preset.lmcache_port)
        env["TF_GLM53_LMCACHE_PORT"] = str(preset.lmcache_port)
        env["TF_GLM53_FLUSH_FILE"] = "/rank-logs/.flush-gpu-prefix"
    for _k in ("TF_GLM53_DCP_DECODE_A2A", "NCCL_PROTO", "NCCL_MIN_NCHANNELS",
               "NCCL_MAX_NCHANNELS", "NCCL_BUFFSIZE", "NCCL_NCHANNELS_PER_NET_PEER",
               "TF_GLM53_PROMPT_ROWS", "TF_GLM53_DCP_DECODE_PACKED",
               "TF_GLM53_DSPARK_DEPTH", "TF_GLM53_DSPARK_CONFIDENCE",
               "TF_GLM53_VERIFY_ROWS", "TF_GLM53_PROMPT_SP",
               "TF_GLM53_QUICK_ROWS", "TF_GLM53_UNPACK_MB", "TF_GLM53_PROMPT_REUSE",
               "TF_GLM53_REUSE_STRIDE"):
        if _k in os.environ:
            env[_k] = os.environ[_k]
    flags = [arg for key, value in env.items() for arg in ("-e", f"{key}={value}")]
    preset.rank_logs.mkdir(mode=0o700, exist_ok=True)
    run("docker", "run", "-d", "--name", preset.container, "--init", "--network", "host", "--shm-size", "2g",
        "--gpus", (('"device=%s"' % os.environ["TF_SM120_GPUS"]) if os.environ.get("TF_SM120_GPUS") else "all"),
        "-v", f"{ROOT}:/port:ro", "-v", f"{MODEL}:/model:ro",
        "-v", f"{preset.rank_logs}:/rank-logs",
        *flags, "--entrypoint", "/bin/bash", IMAGE, "/port/run-ranks.sh", str(PORT),
        str(preset.context), str(preset.parallel))


def ready(preset=PRESETS["dcp1"]):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        status = run("docker", "inspect", "--format", "{{.State.Status}}", preset.container, capture=True).strip()
        if status != "running":
            raise RuntimeError(f"{preset.container} is {status}")
        try:
            with opener.open(f"http://127.0.0.1:{PORT}/v1/models", timeout=3) as response:
                if not any(model["id"] == "model" and model["context_length"] == preset.context
                           for model in json.load(response)["data"]):
                    raise RuntimeError("Unexpected served model name or context length")
        except OSError:
            time.sleep(10)
            continue
        for rank in range(4):
            logs = (preset.rank_logs / f"rank-{rank}.log").read_text(errors="replace")
            if f"glm53 rank {rank}: CUDA device {rank}, sm_120" not in logs:
                raise RuntimeError(f"Rank {rank} did not bind its SM120 device")
            if not re.search(rf"glm53 rank {rank}: loaded.*[1-9]\d* decode graphs captured", logs):
                raise RuntimeError(f"CUDA Graph capture did not complete on rank {rank}")
            if preset.gather_prefill and f"rank {rank}: prefill gathers compressed KV" not in logs:
                raise RuntimeError(f"Rank {rank} did not enable gathered prefill")
            if preset.fp8_kv and f"rank {rank}: E4M3 MLA KV" not in logs:
                raise RuntimeError(f"Rank {rank} did not enable FP8 KV")
            if preset.shared_kv and f"{preset.context} tokens in one shared KV pool" not in logs:
                raise RuntimeError(f"Rank {rank} did not enable the shared KV pool")
            if preset.dspark and f"glm53 rank {rank}: DSpark drafter " not in logs:
                raise RuntimeError(f"Rank {rank} did not load DSpark")
            if rank == 0 and preset.lmcache_port and "LMCache: storing learned state payloads" not in logs:
                raise RuntimeError("Rank 0 did not enable LMCache")
            if rank == 0 and f"DCP {preset.dcp}, {preset.parallel} concurrent stream" not in logs:
                raise RuntimeError("Unexpected decode parallelism or concurrency")
        print("READY, CUDA Graphs verified on all four ranks", flush=True)
        return
    raise TimeoutError("Native endpoint did not become ready")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("start", "ready"))
    parser.add_argument("--preset", choices=PRESETS, default="dcp4-fp8-shared")
    args = parser.parse_args()
    preset = PRESETS[args.preset]
    if args.command == "start":
        start(preflight(preset), preset)
    ready(preset)
