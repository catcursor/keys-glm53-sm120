"""Serve native TensorFold state chunks through LMCache, with RAM L1 and disk L2.

L2 is write-through: evicting a chunk from RAM does not discard its disk copy.
Each tier has an independent LRU budget. Entries are complete 4 MiB protocol
chunks, not complete prompts; losing any chunk causes a native cache miss.
"""

import argparse
from collections import Counter, OrderedDict
import hashlib
import json
import logging
import os
from pathlib import Path
import re
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


class BoundedCache(LMSLocalBackend):
    def __init__(self, capacity, disk_path=None, disk_capacity=0):
        super().__init__()
        if capacity <= 0 or disk_capacity < 0:
            raise ValueError("Cache capacities must be positive (or zero to disable L2)")
        if bool(disk_path) != bool(disk_capacity):
            raise ValueError("Disk path and disk capacity must be specified together")
        self.capacity = capacity
        self.used = 0
        self.disk_capacity = disk_capacity
        self.disk_used = 0
        self.disk = OrderedDict()
        self.counters = Counter()
        self.instance_id = f"{time.time_ns():x}-{os.getpid()}"
        self.db = None
        self.disk_path = None
        self.closed = False
        if disk_path is not None:
            self.disk_path = Path(disk_path).resolve()
            self.disk_path.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.objects = self.disk_path / "objects"
            self.objects.mkdir(mode=0o700, exist_ok=True)
            # An exclusive connection prevents two cache servers from owning
            # the same payload directory. WAL is unnecessary for one owner.
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
        """Reconcile committed records with files; discard interrupted writes."""
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
                    and path.is_file()
                    and not path.is_symlink()
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
        # Only files in this backend's own objects namespace are removed.
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
            self._drop_disk(next(iter(self.disk)), eviction=True)
        self.counters["recovered_chunks"] = len(self.disk)

    def _put_l1(self, key, obj):
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

    def _drop_disk(self, text, eviction=False):
        entry = self.disk.pop(text, None)
        if entry is None:
            return
        self.disk_used -= entry[0]
        # Invalidate the live entry before I/O: an unsuccessful overwrite must
        # never expose an older value after the new RAM value is evicted.
        try:
            self.db.execute("DELETE FROM chunks WHERE key=?", (text,))
            self.db.commit()
        finally:
            self._blob_path(text).unlink(missing_ok=True)
        if eviction:
            self.counters["l2_evictions"] += 1

    def _put_disk(self, text, data):
        self._drop_disk(text)
        if len(data) > self.disk_capacity:
            return
        while self.disk_used + len(data) > self.disk_capacity:
            self._drop_disk(next(iter(self.disk)), eviction=True)
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
                "INSERT INTO chunks VALUES (?, ?, ?, ?)",
                (text, len(data), digest, time.time_ns()),
            )
            self.db.commit()
            self.disk[text] = (len(data), digest)
            self.disk_used += len(data)
            self.counters["l2_write_bytes"] += len(data)
        except Exception:
            self.db.rollback()
            path.unlink(missing_ok=True)
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def put(self, meta, data):
        if (
            data is None
            or len(data) != meta.length
            or not 0 < meta.length <= CHUNK_BYTES
            or meta.fmt != MemoryFormat.BINARY_BUFFER
            or meta.dtype != torch.uint8
            or tuple(meta.shape) != (meta.length, 0, 0, 0)
            or not self._valid_key(meta.key)
        ):
            with self.lock:
                self.counters["rejected_puts"] += 1
            return
        # Immutable bytes also prevent callers from changing a cached chunk.
        data = bytes(data)
        with self.lock:
            self._put_l1(
                meta.key, LMSMemoryObj(data, meta.length, meta.fmt, meta.dtype, meta.shape)
            )
            self.counters["puts"] += 1
            self.counters["put_bytes"] += meta.length
            if self.db is not None:
                try:
                    self._put_disk(meta.key.to_string(), data)
                except (OSError, sqlite3.Error) as error:
                    self.counters["l2_write_errors"] += 1
                    LOG.error("LMCache L2 write failed; RAM copy retained: %s", error)

    def get(self, key):
        with self.lock:
            obj = self.dict.get(key)
            if obj is not None:
                self.dict.move_to_end(key)
                self.counters["l1_hits"] += 1
                self.counters["l1_read_bytes"] += obj.length
                return obj
            self.counters["l1_misses"] += 1
            text = key.to_string()
            entry = self.disk.get(text)
            if entry is not None:
                try:
                    path = self._blob_path(text)
                    if path.is_symlink() or path.stat().st_size != entry[0]:
                        raise ValueError("L2 payload length changed")
                    data = path.read_bytes()
                    if len(data) != entry[0] or hashlib.sha256(data).hexdigest() != entry[1]:
                        raise ValueError("L2 payload checksum mismatch")
                except (OSError, ValueError) as error:
                    self.counters["l2_read_errors"] += 1
                    LOG.warning("Discarding invalid LMCache L2 chunk: %s", error)
                    self._drop_disk(text)
                else:
                    obj = LMSMemoryObj(
                        data, len(data), MemoryFormat.BINARY_BUFFER,
                        torch.uint8, torch.Size((len(data), 0, 0, 0)),
                    )
                    self.disk.move_to_end(text)
                    self._put_l1(key, obj)
                    self.counters["l2_hits"] += 1
                    self.counters["l2_read_bytes"] += len(data)
                    return obj
            self.counters["misses"] += 1
            return None

    def contains(self, key):
        with self.lock:
            self.counters["contains"] += 1
            if key in self.dict:
                return True
            text = key.to_string()
            entry = self.disk.get(text)
            if entry is None:
                return False
            path = self._blob_path(text)
            try:
                valid = not path.is_symlink() and path.stat().st_size == entry[0]
            except OSError:
                valid = False
            if not valid:
                self._drop_disk(text)
            return valid

    def list_keys(self):
        with self.lock:
            keys = {key.to_string(): key for key in self.dict}
            for text in self.disk:
                if text not in keys:
                    keys[text] = parse_cache_key(text)
            return list(keys.values())

    def remove(self, key):
        with self.lock:
            obj = self.dict.pop(key, None)
            if obj is not None:
                self.used -= obj.length
            if self.db is not None:
                self._drop_disk(key.to_string())

    def snapshot(self):
        with self.lock:
            return {
                "time": time.time(), "process_pid": os.getpid(),
                "instance_id": self.instance_id,
                "l1_capacity_bytes": self.capacity,
                "l1_used_bytes": self.used, "l1_chunks": len(self.dict),
                "l2_capacity_bytes": self.disk_capacity,
                "l2_used_bytes": self.disk_used, "l2_chunks": len(self.disk),
                "l2_path": str(self.disk_path) if self.disk_path else None,
                "counters": dict(self.counters),
            }

    def close(self):
        with self.lock:
            if self.closed:
                return
            if self.db is not None:
                # Persist current LRU order on a graceful shutdown. After an
                # abrupt stop the insertion order remains a valid fallback.
                base = time.time_ns()
                self.db.executemany(
                    "UPDATE chunks SET touched=? WHERE key=?",
                    ((base + index, text) for index, text in enumerate(self.disk)),
                )
                self.db.commit()
                self.db.close()
                self.db = None
            self.closed = True


class TensorFoldCacheServer(LMCacheServer):
    def receive_all(self, sock, n):
        if not 0 < n <= CHUNK_BYTES:
            raise ValueError("Invalid LMCache frame size")
        return super().receive_all(sock, n)

    def handle_client(self, sock):
        sock.settimeout(10)
        super().handle_client(sock)


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
    parser.add_argument("--stats-path", type=Path)
    args = parser.parse_args()
    if not 0 < args.port < 65536:
        parser.error("--port must be between 1 and 65535")
    if args.disk_gib < 0 or bool(args.disk_path) != bool(args.disk_gib):
        parser.error("--disk-path and a positive --disk-gib must be specified together")
    cache = BoundedCache(args.gib << 30, args.disk_path, args.disk_gib << 30)
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
        f"RAM L1 {args.gib} GiB, disk L2 {args.disk_gib} GiB at {args.disk_path}",
        flush=True,
    )
    try:
        server.run()
    finally:
        stop.set()
        if reporter is not None:
            reporter.join()
            write_stats(cache, args.stats_path)
        cache.close()


if __name__ == "__main__":
    main()
