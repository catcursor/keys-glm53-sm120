"""Serve native TensorFold state chunks through LMCache, with RAM L1 and disk L2.

L2 is a bounded background write-back cache. PUT acknowledges RAM availability;
only completed L2 writes survive a forced exit. Entries are complete 4 MiB
protocol chunks, not complete prompts; losing any chunk causes a native miss.
"""

import argparse
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import socket
import sqlite3
import tempfile
import threading
import time

import torch

from lmcache.utils import parse_cache_key
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.server.__main__ import LMCacheServer
from lmcache.v1.server.storage_backend.local_backend import LMSLocalBackend
from lmcache.v1.server.utils import LMSMemoryObj

CHUNK_BYTES = 4 << 20
LOG = logging.getLogger(__name__)


@dataclass(eq=False)
class WriteJob:
    text: str
    obj: LMSMemoryObj
    cancelled: bool = False


class BoundedCache(LMSLocalBackend):
    def __init__(self, capacity, disk_path=None, disk_capacity=0,
                 queue_capacity=512 << 20, queue_wait_seconds=1.0):
        super().__init__()
        if capacity <= 0 or disk_capacity < 0 or queue_capacity <= 0:
            raise ValueError("Invalid cache or write-queue capacity")
        if bool(disk_path) != bool(disk_capacity):
            raise ValueError("Disk path and disk capacity must be specified together")
        if not math.isfinite(queue_wait_seconds) or queue_wait_seconds < 0:
            raise ValueError("Queue wait must be finite and non-negative")
        self.capacity, self.used = capacity, 0
        self.disk_capacity, self.disk_used = disk_capacity, 0
        self.queue_capacity = queue_capacity
        self.queue_chunk_limit = max(1, queue_capacity // CHUNK_BYTES)
        self.queue_wait_seconds = queue_wait_seconds
        self.waiting_puts = 0
        self.pending_bytes = 0  # Includes the writer's active immutable payload.
        self.jobs = deque()
        self.pending = {}
        self.active = None
        self.retired = OrderedDict()
        self.active_retired = None
        self.disk = OrderedDict()
        self.orphan_bytes = 0
        self.counters = Counter()
        self.instance_id = f"{time.time_ns():x}-{os.getpid()}"
        self.db = None
        self.disk_path = None
        self.closing = self.closed = False
        self.cv = threading.Condition(self.lock)
        self.io_lock = threading.Lock()
        self.writer = None
        if disk_path is not None:
            self.disk_path = Path(disk_path).resolve()
            self.disk_path.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.objects = self.disk_path / "objects"
            self.objects.mkdir(mode=0o700, exist_ok=True)
            self.db = sqlite3.connect(
                self.disk_path / "index.sqlite3", timeout=0, check_same_thread=False
            )
            try:
                self.db.execute("PRAGMA locking_mode=EXCLUSIVE")
                self.db.execute("PRAGMA synchronous=FULL")
                self.db.execute("BEGIN EXCLUSIVE")
                version = self.db.execute("PRAGMA user_version").fetchone()[0]
                if version not in (0, 1):
                    raise ValueError(f"Unsupported cache index version: {version}")
                self.db.execute(
                    "CREATE TABLE IF NOT EXISTS chunks "
                    "(key TEXT PRIMARY KEY, length INTEGER NOT NULL, "
                    "digest TEXT NOT NULL, touched INTEGER NOT NULL)"
                )
                self.db.execute("PRAGMA user_version=1")
                self.db.commit()
                self._recover_disk()
            except Exception:
                self.db.close()
                self.db = None
                raise
            self.writer = threading.Thread(
                target=self._writer_loop, name="lmcache-l2-writer", daemon=True
            )
            self.writer.start()

    @staticmethod
    def _valid_key(key):
        return (
            key.model_name.startswith("tf-glm53-v1-")
            and key.dtype == torch.uint8
            and len(key.to_string()) <= 150
        )

    def _blob_path(self, text):
        name = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return self.objects / name[:2] / (name + ".bin")

    def _recover_disk(self):
        """Reconcile committed records before accepting requests."""
        referenced = set()
        for text, length, digest in self.db.execute(
            "SELECT key, length, digest FROM chunks ORDER BY touched, key"
        ).fetchall():
            path = self._blob_path(text)
            try:
                valid = (
                    self._valid_key(parse_cache_key(text))
                    and 0 < length <= CHUNK_BYTES
                    and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                    and path.is_file() and not path.is_symlink()
                    and path.stat().st_size == length
                )
            except (OSError, ValueError, KeyError, TypeError):
                valid = False
            if valid:
                self.disk[text] = (length, digest)
                self.disk_used += length
                referenced.add(path)
            else:
                self.db.execute("DELETE FROM chunks WHERE key=?", (text,))
                self.counters["recovery_discarded"] += 1
        self.db.commit()
        for directory in self.objects.iterdir():
            if not directory.is_dir() or directory.is_symlink():
                continue
            if re.fullmatch(r"[0-9a-f]{2}", directory.name) is None:
                continue
            for path in directory.iterdir():
                own_file = re.fullmatch(r"[0-9a-f]{64}\.bin", path.name)
                own_temp = path.name.startswith(".write-") and path.name.endswith(".tmp")
                if (own_file or own_temp) and path not in referenced:
                    path.unlink(missing_ok=True)
                    self.counters["recovery_orphans"] += 1
        while self.disk_used > self.disk_capacity:
            text, entry = self.disk.popitem(last=False)
            self._discard_blob_io(text, entry)
            self.counters["l2_evictions"] += 1
        self.counters["recovered_chunks"] = len(self.disk)

    def _put_l1_locked(self, key, obj):
        old = self.dict.pop(key, None)
        if old is not None:
            self.used -= old.length
        if obj.length > self.capacity:
            return
        while self.used + obj.length > self.capacity:
            _, evicted = self.dict.popitem(last=False)
            self.used -= evicted.length
            self.counters["l1_evictions"] += 1
        self.dict[key] = obj
        self.used += obj.length

    def _retire_locked(self, text):
        entry = self.disk.pop(text, None)
        if entry is not None:
            self.retired[text] = entry
            self.counters["l2_invalidations"] += 1
            self.cv.notify_all()

    def _cancel_locked(self, text):
        job = self.pending.pop(text, None)
        if job is not None:
            job.cancelled = True
            if job is not self.active:
                self.jobs.remove(job)
                self.pending_bytes -= job.obj.length
            self.counters["l2_superseded"] += 1

    def _discard_blob_io(self, text, entry, accounted=True):
        """Caller owns io_lock, or is the constructor before the writer starts."""
        try:
            self.db.execute("DELETE FROM chunks WHERE key=?", (text,))
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        finally:
            try:
                self._blob_path(text).unlink(missing_ok=True)
            except OSError:
                with self.lock:
                    self.orphan_bytes += entry[0]
                    self.counters["l2_cleanup_errors"] += 1
                raise
            finally:
                if accounted:
                    with self.lock:
                        self.disk_used -= entry[0]

    def _write_payload_io(self, text, data):
        """Durable payload/index write, deliberately outside the RAM lock."""
        path = self._blob_path(text)
        path.parent.mkdir(mode=0o700, exist_ok=True)
        temporary = None
        digest = hashlib.sha256(data).hexdigest()
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=".write-", suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            temporary = None
            if hasattr(os, "O_DIRECTORY"):
                fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            self.db.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?)",
                (text, len(data), digest, time.time_ns()),
            )
            self.db.commit()
            return (len(data), digest)
        except Exception:
            self.db.rollback()
            path.unlink(missing_ok=True)
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _write_job_io(self, job):
        with self.lock:
            if job.cancelled or self.pending.get(job.text) is not job:
                return
        if job.obj.length > self.disk_capacity:
            with self.lock:
                self.counters["l2_oversize_skips"] += 1
            return
        # Retired bytes still count until their files have actually been removed.
        while True:
            with self.lock:
                if self.disk_used + self.orphan_bytes + job.obj.length <= self.disk_capacity:
                    break
                if self.retired:
                    text, entry = self.retired.popitem(last=False)
                elif self.disk:
                    text, entry = self.disk.popitem(last=False)
                    self.counters["l2_evictions"] += 1
                else:
                    self.counters["l2_capacity_skips"] += 1
                    return
            self._discard_blob_io(text, entry)
        entry = self._write_payload_io(job.text, job.obj.data)
        with self.lock:
            current = not job.cancelled and self.pending.get(job.text) is job
            if current:
                self.disk[job.text] = entry
                self.disk_used += entry[0]
                self.counters["l2_write_bytes"] += entry[0]
                self.counters["l2_completed"] += 1
        if not current:
            # A PUT/remove arrived during fsync. Never publish the older value.
            self._discard_blob_io(job.text, entry, accounted=False)

    def _writer_loop(self):
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.retired or self.jobs or self.closing)
                if self.retired:
                    retired = self.retired.popitem(last=False)
                    self.active_retired = retired
                    job = None
                elif self.jobs:
                    job = self.jobs.popleft()
                    self.active = job
                    retired = None
                else:
                    return
            try:
                with self.io_lock:
                    if retired is not None:
                        self._discard_blob_io(*retired)
                    else:
                        self._write_job_io(job)
            except Exception as error:
                with self.lock:
                    self.counters["l2_write_errors"] += 1
                LOG.error("LMCache background L2 operation failed: %s", error)
            finally:
                with self.cv:
                    if job is not None:
                        self.pending_bytes -= job.obj.length
                        if self.pending.get(job.text) is job:
                            self.pending.pop(job.text)
                        self.active = None
                    self.active_retired = None
                    self.cv.notify_all()
                job = retired = None

    def put(self, meta, data):
        if (
            data is None or len(data) != meta.length
            or not 0 < meta.length <= CHUNK_BYTES
            or meta.fmt != MemoryFormat.BINARY_BUFFER or meta.dtype != torch.uint8
            or tuple(meta.shape) != (meta.length, 0, 0, 0)
            or not self._valid_key(meta.key)
        ):
            with self.lock:
                self.counters["rejected_puts"] += 1
            return False
        obj = LMSMemoryObj(bytes(data), meta.length, meta.fmt, meta.dtype, meta.shape)
        text = meta.key.to_string()
        with self.cv:
            if self.closing:
                self.counters["closing_puts"] += 1
                return False
            self._put_l1_locked(meta.key, obj)
            self.counters["puts"] += 1
            self.counters["put_bytes"] += meta.length
            if self.db is not None:
                self._cancel_locked(text)
                self._retire_locked(text)

                def available():
                    return (
                        self.pending_bytes + obj.length <= self.queue_capacity
                        and len(self.jobs) + (self.active is not None) < self.queue_chunk_limit
                    )

                if not available() and self.queue_wait_seconds:
                    started = time.monotonic_ns()
                    self.waiting_puts += 1
                    self.counters["l2_backpressure_waits"] += 1
                    # Only this upload connection waits. Condition.wait releases
                    # the RAM lock, so GET, stats and other connections continue.
                    self.cv.notify_all()
                    try:
                        self.cv.wait_for(
                            lambda: self.closing or self.dict.get(meta.key) is not obj or available(),
                            timeout=self.queue_wait_seconds,
                        )
                    finally:
                        self.waiting_puts -= 1
                        self.counters["l2_backpressure_wait_ns"] += time.monotonic_ns() - started
                        self.cv.notify_all()
                    if self.closing:
                        self.counters["l2_wait_cancelled"] += 1
                        return False
                    if self.dict.get(meta.key) is not obj:
                        self.counters["l2_wait_superseded"] += 1
                        return False
                    if not available():
                        self.counters["l2_backpressure_timeouts"] += 1
                if available():
                    job = WriteJob(text, obj)
                    self.pending[text] = job
                    self.jobs.append(job)
                    self.pending_bytes += obj.length
                    self.counters["l2_enqueued"] += 1
                    self.counters["l2_queue_peak_bytes"] = max(
                        self.counters["l2_queue_peak_bytes"], self.pending_bytes
                    )
                else:
                    self.counters["l2_queue_full"] += 1
                    self.counters["l2_skipped_bytes"] += obj.length
                self.cv.notify_all()
            # Availability is best effort; it is not an L2 durability promise.
            return meta.key in self.dict or text in self.pending

    def get(self, key):
        text = key.to_string()
        with self.lock:
            obj = self.dict.get(key)
            if obj is not None:
                self.dict.move_to_end(key)
                self.counters["l1_hits"] += 1
                self.counters["l1_read_bytes"] += obj.length
                return obj
            self.counters["l1_misses"] += 1
            job = self.pending.get(text)
            if job is not None and not job.cancelled:
                self.counters["pending_hits"] += 1
                self._put_l1_locked(key, job.obj)
                return job.obj
            entry = self.disk.get(text)
            if entry is None or self.closed:
                self.counters["misses"] += 1
                return None
        # L2 reads may serialize with L2 writes, but neither blocks RAM traffic.
        with self.io_lock:
            with self.lock:
                if self.disk.get(text) is not entry or self.closed:
                    self.counters["misses"] += 1
                    return None
            try:
                path = self._blob_path(text)
                if path.is_symlink() or path.stat().st_size != entry[0]:
                    raise ValueError("L2 payload length changed")
                data = path.read_bytes()
                if len(data) != entry[0] or hashlib.sha256(data).hexdigest() != entry[1]:
                    raise ValueError("L2 payload checksum mismatch")
            except (OSError, ValueError) as error:
                with self.cv:
                    self.counters["l2_read_errors"] += 1
                    self.counters["misses"] += 1
                    if self.disk.get(text) is entry:
                        self._retire_locked(text)
                LOG.warning("Discarding invalid LMCache L2 chunk: %s", error)
                return None
            with self.lock:
                if self.disk.get(text) is not entry:
                    self.counters["misses"] += 1
                    return None
                obj = LMSMemoryObj(
                    data, len(data), MemoryFormat.BINARY_BUFFER,
                    torch.uint8, torch.Size((len(data), 0, 0, 0)),
                )
                self.disk.move_to_end(text)
                self._put_l1_locked(key, obj)
                self.counters["l2_hits"] += 1
                self.counters["l2_read_bytes"] += len(data)
                return obj

    def contains(self, key):
        text = key.to_string()
        with self.lock:
            self.counters["contains"] += 1
            if key in self.dict or text in self.pending:
                return True
            entry = self.disk.get(text)
            if entry is None or self.closed:
                return False
        try:
            path = self._blob_path(text)
            valid = not path.is_symlink() and path.stat().st_size == entry[0]
        except OSError:
            valid = False
        with self.cv:
            if self.disk.get(text) is not entry:
                return False
            if not valid:
                self._retire_locked(text)
            return valid

    def list_keys(self):
        with self.lock:
            keys = {key.to_string(): key for key in self.dict}
            for text in self.disk.keys() | self.pending.keys():
                if text not in keys:
                    keys[text] = parse_cache_key(text)
            return list(keys.values())

    def remove(self, key):
        with self.cv:
            obj = self.dict.pop(key, None)
            if obj is not None:
                self.used -= obj.length
            text = key.to_string()
            self._cancel_locked(text)
            self._retire_locked(text)
            self.cv.notify_all()

    def snapshot(self):
        with self.lock:
            return {
                "time": time.time(), "process_pid": os.getpid(),
                "instance_id": self.instance_id,
                "l1_capacity_bytes": self.capacity, "l1_used_bytes": self.used,
                "l1_chunks": len(self.dict), "l2_capacity_bytes": self.disk_capacity,
                "l2_used_bytes": self.disk_used, "l2_chunks": len(self.disk),
                "l2_path": str(self.disk_path) if self.disk_path else None,
                "l2_pending_bytes": self.pending_bytes,
                "l2_pending_chunks": len(self.jobs) + (self.active is not None),
                "l2_queue_capacity_bytes": self.queue_capacity,
                "l2_queue_chunk_limit": self.queue_chunk_limit,
                "l2_queue_wait_seconds": self.queue_wait_seconds,
                "l2_waiting_puts": self.waiting_puts,
                "l2_retired_chunks": len(self.retired) + (self.active_retired is not None),
                "l2_orphan_bytes": self.orphan_bytes,
                "closing": self.closing, "counters": dict(self.counters),
            }

    def flush(self, timeout=None):
        """Wait for accepted writes and invalidations; inspect counters for errors."""
        with self.cv:
            return self.cv.wait_for(
                lambda: not (
                    self.jobs or self.active or self.retired
                    or self.active_retired or self.waiting_puts
                ),
                timeout=timeout,
            )

    def close(self):
        with self.cv:
            if self.closed:
                return
            self.closing = True
            self.cv.notify_all()
        if self.writer is not None:
            self.writer.join()
        with self.io_lock:
            if self.db is not None:
                with self.lock:
                    order = list(self.disk)
                base = time.time_ns()
                self.db.executemany(
                    "UPDATE chunks SET touched=? WHERE key=?",
                    ((base + index, text) for index, text in enumerate(order)),
                )
                self.db.commit()
                self.db.close()
                self.db = None
            with self.lock:
                self.closed = True

class TensorFoldCacheServer(LMCacheServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clients = set()
        self.clients_lock = threading.Lock()
        self.stopping = False

    def receive_all(self, sock, n):
        if not 0 < n <= CHUNK_BYTES:
            raise ValueError("Invalid LMCache frame size")
        return super().receive_all(sock, n)

    def handle_client(self, sock):
        with self.clients_lock:
            if self.stopping:
                sock.close()
                return
            self.clients.add(sock)
        try:
            sock.settimeout(10)
            super().handle_client(sock)
        except OSError:
            if not self.stopping:
                raise
        finally:
            with self.clients_lock:
                self.clients.discard(sock)

    def stop_clients(self):
        with self.clients_lock:
            self.stopping = True
            clients = list(self.clients)
        for sock in clients:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def write_stats(cache, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(cache.snapshot(), indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def report_stats(cache, path, stop):
    while not stop.wait(5):
        try:
            write_stats(cache, path)
        except OSError as error:
            LOG.error("Cannot write LMCache statistics: %s", error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gib", type=int, default=16, choices=range(1, 65))
    parser.add_argument("--disk-gib", type=int, default=0)
    parser.add_argument("--disk-path", type=Path)
    parser.add_argument("--write-queue-mib", type=int, default=512)
    parser.add_argument("--write-queue-wait-ms", type=int, default=1000)
    parser.add_argument("--stats-path", type=Path)
    args = parser.parse_args()
    if not 0 < args.port < 65536:
        parser.error("--port must be between 1 and 65535")
    if args.disk_gib < 0 or bool(args.disk_path) != bool(args.disk_gib):
        parser.error("--disk-path and a positive --disk-gib must be specified together")
    if args.write_queue_mib < 4:
        parser.error("--write-queue-mib must be at least 4")
    if args.write_queue_wait_ms < 0:
        parser.error("--write-queue-wait-ms must be non-negative")
    cache = BoundedCache(
        args.gib << 30, args.disk_path, args.disk_gib << 30,
        queue_capacity=args.write_queue_mib << 20,
        queue_wait_seconds=args.write_queue_wait_ms / 1000,
    )
    server = TensorFoldCacheServer("127.0.0.1", args.port, "cpu")
    server.data_store.close()
    server.data_store = cache
    stop = threading.Event()
    reporter = None
    if args.stats_path:
        write_stats(cache, args.stats_path)
        reporter = threading.Thread(
            target=report_stats, args=(cache, args.stats_path, stop), daemon=True,
        )
        reporter.start()
    print(
        f"TensorFold LMCache listening on 127.0.0.1:{args.port}: "
        f"RAM L1 {args.gib} GiB, disk L2 {args.disk_gib} GiB at {args.disk_path}; "
        f"background write queue {args.write_queue_mib} MiB, "
        f"upload backpressure at most {args.write_queue_wait_ms} ms",
        flush=True,
    )
    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop_clients()
        stop.set()
        if reporter is not None:
            reporter.join()
        cache.close()
        if args.stats_path:
            write_stats(cache, args.stats_path)


if __name__ == "__main__":
    main()
