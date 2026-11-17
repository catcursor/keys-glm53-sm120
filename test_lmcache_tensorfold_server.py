"""Run in the pinned LMCache image; no model or GPU is needed."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import struct
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from lmcache.utils import parse_cache_key
from lmcache.v1.memory_management import MemoryFormat

from lmcache_tensorfold_server import (
    BoundedCache, TensorFoldCacheServer, write_stats,
)


def key(index):
    return parse_cache_key(f"tf-glm53-v1-123456789abcdef0@4@0@{index:x}@uint8")


def meta(index, data):
    return SimpleNamespace(
        key=key(index), length=len(data), fmt=MemoryFormat.BINARY_BUFFER,
        dtype=torch.uint8, shape=torch.Size((len(data), 0, 0, 0)),
    )


class TieredCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def cache(self, ram=8, disk=32):
        cache = BoundedCache(ram, self.root if disk else None, disk)
        self.addCleanup(cache.close)
        return cache

    def test_ram_lru_is_independent_from_persistent_disk(self):
        cache = self.cache()
        cache.put(meta(1, b"1111"), b"1111")
        cache.put(meta(2, b"2222"), b"2222")
        self.assertEqual(cache.get(key(1)).data, b"1111")
        cache.put(meta(3, b"3333"), b"3333")
        self.assertNotIn(key(2), cache.dict)
        self.assertTrue(cache.contains(key(2)))
        self.assertEqual(cache.get(key(2)).data, b"2222")
        self.assertEqual(cache.snapshot()["counters"]["l2_hits"], 1)
        self.assertEqual(len(cache.list_keys()), 3)

    def test_disk_restart_preserves_payload_and_metadata(self):
        cache = self.cache()
        cache.put(meta(1, b"12345678"), b"12345678")
        cache.close()
        cache = self.cache()
        self.assertEqual(cache.snapshot()["l1_used_bytes"], 0)
        self.assertEqual(cache.snapshot()["counters"]["recovered_chunks"], 1)
        self.assertTrue(cache.contains(key(1)))
        obj = cache.get(key(1))
        self.assertEqual(obj.data, b"12345678")
        self.assertEqual(obj.dtype, torch.uint8)
        self.assertEqual(obj.fmt, MemoryFormat.BINARY_BUFFER)
        self.assertEqual(tuple(obj.shape), (8, 0, 0, 0))

    def test_l2_capacity_evicts_lru_not_l1(self):
        cache = self.cache(ram=16, disk=8)
        for i in range(1, 4):
            data = bytes([i]) * 4
            cache.put(meta(i, data), data)
        self.assertEqual(cache.snapshot()["l2_used_bytes"], 8)
        self.assertEqual(cache.get(key(1)).data, b"\x01" * 4)
        cache.close()
        cache = self.cache(ram=16, disk=8)
        self.assertFalse(cache.contains(key(1)))
        self.assertIsNone(cache.get(key(1)))
        self.assertEqual(cache.get(key(3)).data, b"\x03" * 4)

    def test_corrupt_disk_chunk_is_a_miss_never_wrong_data(self):
        cache = self.cache(ram=4)
        cache.put(meta(1, b"1111"), b"1111")
        cache.put(meta(2, b"2222"), b"2222")
        cache._blob_path(key(1).to_string()).write_bytes(b"BAD!")
        self.assertIsNone(cache.get(key(1)))
        self.assertFalse(cache.contains(key(1)))
        self.assertEqual(cache.snapshot()["counters"]["l2_read_errors"], 1)

    def test_missing_truncated_and_orphan_files_reconcile(self):
        cache = self.cache()
        cache.put(meta(1, b"1111"), b"1111")
        cache.put(meta(2, b"2222"), b"2222")
        missing = cache._blob_path(key(1).to_string())
        truncated = cache._blob_path(key(2).to_string())
        orphan = cache._blob_path(key(3).to_string())
        cache.close()
        missing.unlink()
        truncated.write_bytes(b"2")
        orphan.parent.mkdir(exist_ok=True)
        orphan.write_bytes(b"3333")
        interrupted = orphan.parent / ".write-abcd.tmp"
        interrupted.write_bytes(b"unfinished")
        cache = self.cache()
        self.assertEqual(cache.snapshot()["l2_used_bytes"], 0)
        self.assertFalse(orphan.exists())
        self.assertFalse(interrupted.exists())
        self.assertFalse(cache.contains(key(1)))
        self.assertFalse(cache.contains(key(2)))

    def test_overwrite_remove_and_caller_mutation(self):
        cache = self.cache()
        data = bytearray(b"1111")
        cache.put(meta(1, data), data)
        data[:] = b"BAD!"
        self.assertEqual(cache.get(key(1)).data, b"1111")
        cache.put(meta(1, b"new"), b"new")
        self.assertEqual(cache.snapshot()["l1_used_bytes"], 3)
        self.assertEqual(cache.snapshot()["l2_used_bytes"], 3)
        cache.close()
        cache = self.cache()
        self.assertEqual(cache.get(key(1)).data, b"new")
        cache.remove(key(1))
        cache.remove(key(1))
        self.assertEqual(cache.snapshot()["l1_used_bytes"], 0)
        self.assertEqual(cache.snapshot()["l2_used_bytes"], 0)
        self.assertFalse(cache.contains(key(1)))

    def test_invalid_chunks_are_rejected_and_ram_only_works(self):
        cache = self.cache(ram=8, disk=0)
        bad = meta(1, b"abc")
        bad.shape = torch.Size((3, 1, 0, 0))
        cache.put(bad, b"abc")
        cache.put(meta(2, b"abcd"), b"ab")
        cache.put(meta(3, b"abcd"), None)
        self.assertEqual(len(cache.list_keys()), 0)
        cache.put(meta(4, b"abcd"), b"abcd")
        self.assertEqual(cache.get(key(4)).data, b"abcd")
        self.assertEqual(cache.snapshot()["counters"]["rejected_puts"], 3)

    def test_failed_disk_overwrite_cannot_return_stale_payload(self):
        cache = self.cache(ram=4)
        cache.put(meta(1, b"old!"), b"old!")
        with patch.object(Path, "unlink", side_effect=PermissionError("injected")):
            cache.put(meta(1, b"new!"), b"new!")
        self.assertEqual(cache.get(key(1)).data, b"new!")
        self.assertEqual(cache.snapshot()["counters"]["l2_write_errors"], 1)
        cache.put(meta(2, b"2222"), b"2222")
        self.assertIsNone(cache.get(key(1)))
        cache.close()
        cache = self.cache(ram=4)
        self.assertIsNone(cache.get(key(1)))

    def test_only_one_process_may_own_disk_index(self):
        cache = self.cache()
        with self.assertRaises(Exception):
            BoundedCache(8, self.root, 32)
        self.assertIsNotNone(cache.db)

    def test_concurrent_rank_writes_respect_budgets(self):
        cache = self.cache(ram=32, disk=128)

        def worker(i):
            data = bytes([i]) * 8
            cache.put(meta(i, data), data)
            result = cache.get(key(i))
            if result is not None:
                self.assertEqual(result.data, data)

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(worker, range(32)))
        snapshot = cache.snapshot()
        self.assertLessEqual(snapshot["l1_used_bytes"], 32)
        self.assertLessEqual(snapshot["l2_used_bytes"], 128)
        self.assertEqual(snapshot["l1_used_bytes"], sum(v.length for v in cache.dict.values()))
        self.assertEqual(snapshot["l2_used_bytes"], sum(v[0] for v in cache.disk.values()))
        stats = self.root / "stats.json"
        write_stats(cache, stats)
        self.assertEqual(json.loads(stats.read_text())["counters"]["puts"], 32)

    def test_native_wire_put_exists_get_and_missing(self):
        cache = self.cache(ram=4)
        server = TensorFoldCacheServer("127.0.0.1", 0, "cpu")
        self.addCleanup(server.server_socket.close)
        server.data_store = cache
        client, peer = socket.socketpair()
        client.settimeout(3)
        thread = threading.Thread(target=server.handle_client, args=(peer,))
        thread.start()

        def request(command, index, data=b""):
            header = struct.pack(
                "<9i150s", command, len(data), 5, 6, 0, len(data), 0, 0, 0,
                key(index).to_string().encode().ljust(150, b" "),
            )
            client.sendall(header + data)

        def receive(n):
            data = bytearray()
            while len(data) < n:
                chunk = client.recv(n - len(data))
                self.assertTrue(chunk)
                data.extend(chunk)
            return bytes(data)

        try:
            request(1, 1, b"1111")
            request(1, 2, b"2222")
            request(3, 1)
            self.assertEqual(struct.unpack("<9i", receive(36))[:2], (200, 0))
            request(2, 1)
            reply = struct.unpack("<9i", receive(36))
            self.assertEqual(reply[:4], (200, 4, 5, 6))
            self.assertEqual(receive(4), b"1111")
            self.assertEqual(cache.snapshot()["counters"]["l2_hits"], 1)
            request(2, 999)
            self.assertEqual(struct.unpack("<9i", receive(36))[:2], (400, 0))
        finally:
            client.close()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
