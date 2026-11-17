"""Run in the pinned LMCache image; no model or GPU is needed."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import struct
import tempfile
import threading
import time
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

    def cache(self, ram=8, disk=32, queue=512 << 20, wait=1.0):
        cache = BoundedCache(
            ram, self.root if disk else None, disk,
            queue_capacity=queue, queue_wait_seconds=wait,
        )
        self.addCleanup(cache.close)
        return cache

    def settle(self, cache):
        self.assertTrue(cache.flush(timeout=5), cache.snapshot())

    def test_ram_lru_is_independent_from_persistent_disk(self):
        cache = self.cache()
        cache.put(meta(1, b"1111"), b"1111")
        cache.put(meta(2, b"2222"), b"2222")
        self.assertEqual(cache.get(key(1)).data, b"1111")
        cache.put(meta(3, b"3333"), b"3333")
        self.settle(cache)
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
        self.settle(cache)
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
        self.settle(cache)
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
        self.settle(cache)
        self.assertEqual(cache.snapshot()["l1_used_bytes"], 3)
        self.assertEqual(cache.snapshot()["l2_used_bytes"], 3)
        cache.close()
        cache = self.cache()
        self.assertEqual(cache.get(key(1)).data, b"new")
        cache.remove(key(1))
        cache.remove(key(1))
        self.settle(cache)
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
        self.settle(cache)
        with patch.object(cache, "_write_payload_io", side_effect=OSError("injected")):
            cache.put(meta(1, b"new!"), b"new!")
            self.settle(cache)
        self.assertEqual(cache.get(key(1)).data, b"new!")
        self.assertEqual(cache.snapshot()["counters"]["l2_write_errors"], 1)
        cache.put(meta(2, b"2222"), b"2222")
        self.settle(cache)
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
        self.settle(cache)
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
            self.settle(cache)
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

    def block_writer(self, cache):
        entered, release = threading.Event(), threading.Event()
        original = cache._write_payload_io

        def blocked(text, data):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test did not release writer")
            return original(text, data)

        return entered, release, patch.object(cache, "_write_payload_io", side_effect=blocked)

    def test_slow_disk_does_not_block_ram_operations(self):
        cache = self.cache(ram=16)
        entered, release, gate = self.block_writer(cache)
        with gate, ThreadPoolExecutor(max_workers=1) as executor:
            try:
                cache.put(meta(1, b"1111"), b"1111")
                self.assertTrue(entered.wait(2))

                def ram_operations():
                    self.assertTrue(cache.put(meta(2, b"2222"), b"2222"))
                    self.assertTrue(cache.contains(key(2)))
                    self.assertEqual(cache.get(key(1)).data, b"1111")
                    return cache.snapshot()

                snapshot = executor.submit(ram_operations).result(timeout=1)
                self.assertGreater(snapshot["l2_pending_bytes"], 0)
            finally:
                release.set()
            self.settle(cache)

    def test_queue_bound_counts_active_payload_and_skips_without_blocking(self):
        cache = self.cache(ram=4, queue=4, wait=0)
        entered, release, gate = self.block_writer(cache)
        with gate:
            try:
                cache.put(meta(1, b"1111"), b"1111")
                self.assertTrue(entered.wait(2))
                self.assertTrue(cache.put(meta(2, b"2222"), b"2222"))
                self.assertTrue(cache.contains(key(2)))
                self.assertEqual(cache.get(key(2)).data, b"2222")
                snapshot = cache.snapshot()
                self.assertEqual(snapshot["l2_pending_bytes"], 4)
                self.assertEqual(snapshot["l2_pending_chunks"], 1)
                self.assertEqual(snapshot["counters"]["l2_queue_full"], 1)
                cache.put(meta(3, b"3333"), b"3333")
                self.assertIsNone(cache.get(key(2)))
            finally:
                release.set()
            self.settle(cache)
        self.assertEqual(cache.get(key(1)).data, b"1111")
        self.assertLessEqual(cache.snapshot()["counters"]["l2_queue_peak_bytes"], 4)

    def test_active_overwrite_publishes_only_latest_value(self):
        cache = self.cache(ram=4)
        entered, release, gate = self.block_writer(cache)
        with gate:
            try:
                cache.put(meta(1, b"old!"), b"old!")
                self.assertTrue(entered.wait(2))
                cache.put(meta(1, b"new!"), b"new!")
                cache.put(meta(2, b"2222"), b"2222")
                self.assertEqual(cache.get(key(1)).data, b"new!")
            finally:
                release.set()
            self.settle(cache)
        cache.close()
        cache = self.cache(ram=4)
        self.assertEqual(cache.get(key(1)).data, b"new!")

    def test_queue_full_overwrite_invalidates_existing_disk_value(self):
        cache = self.cache(ram=4, queue=4, wait=0)
        cache.put(meta(1, b"old!"), b"old!")
        self.settle(cache)
        entered, release, gate = self.block_writer(cache)
        with gate:
            try:
                cache.put(meta(2, b"2222"), b"2222")
                self.assertTrue(entered.wait(2))
                cache.put(meta(1, b"new!"), b"new!")
                self.assertEqual(cache.get(key(1)).data, b"new!")
                cache.put(meta(3, b"3333"), b"3333")
                self.assertIsNone(cache.get(key(1)))
            finally:
                release.set()
            self.settle(cache)
        cache.close()
        cache = self.cache(ram=4, queue=4, wait=0)
        self.assertIsNone(cache.get(key(1)))

    def test_remove_during_write_cannot_resurrect_chunk(self):
        cache = self.cache()
        entered, release, gate = self.block_writer(cache)
        with gate:
            try:
                cache.put(meta(1, b"1111"), b"1111")
                self.assertTrue(entered.wait(2))
                cache.remove(key(1))
                self.assertFalse(cache.contains(key(1)))
            finally:
                release.set()
            self.settle(cache)
        cache.close()
        cache = self.cache()
        self.assertIsNone(cache.get(key(1)))

    def test_overwrite_during_disk_read_never_promotes_old_value(self):
        cache = self.cache(ram=4)
        cache.put(meta(1, b"old!"), b"old!")
        cache.put(meta(2, b"2222"), b"2222")
        self.settle(cache)
        target = cache._blob_path(key(1).to_string())
        entered, release = threading.Event(), threading.Event()
        original = Path.read_bytes

        def blocked(path):
            if path == target:
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release reader")
            return original(path)

        with patch.object(Path, "read_bytes", new=blocked), ThreadPoolExecutor(max_workers=2) as executor:
            try:
                future = executor.submit(cache.get, key(1))
                self.assertTrue(entered.wait(2))
                executor.submit(cache.put, meta(1, b"new!"), b"new!").result(timeout=1)
            finally:
                release.set()
            self.assertIsNone(future.result(timeout=2))
        self.settle(cache)
        self.assertEqual(cache.get(key(1)).data, b"new!")

    def test_close_drains_writes_and_rejects_new_puts(self):
        cache = self.cache()
        entered, release, gate = self.block_writer(cache)
        with gate, ThreadPoolExecutor(max_workers=1) as executor:
            try:
                cache.put(meta(1, b"1111"), b"1111")
                self.assertTrue(entered.wait(2))
                closed = executor.submit(cache.close)
                with self.assertRaises(TimeoutError):
                    closed.result(timeout=0.1)
                self.assertTrue(cache.snapshot()["closing"])
                self.assertFalse(cache.put(meta(2, b"2222"), b"2222"))
            finally:
                release.set()
            closed.result(timeout=5)
        cache = self.cache()
        self.assertEqual(cache.get(key(1)).data, b"1111")
        self.assertFalse(cache.contains(key(2)))

    def wait_for_waiter(self, cache):
        deadline = time.monotonic() + 2
        while cache.snapshot()["l2_waiting_puts"] == 0:
            if time.monotonic() >= deadline:
                self.fail("PUT did not reach backpressure wait")
            time.sleep(0.001)

    def test_backpressure_waits_for_capacity_without_blocking_get(self):
        cache = self.cache(ram=16, queue=4)
        entered, release, gate = self.block_writer(cache)
        with gate, ThreadPoolExecutor(max_workers=2) as executor:
            try:
                cache.put(meta(1, b"1111"), b"1111")
                self.assertTrue(entered.wait(2))
                pending = executor.submit(cache.put, meta(2, b"2222"), b"2222")
                self.wait_for_waiter(cache)
                self.assertFalse(pending.done())
                self.assertEqual(executor.submit(cache.get, key(2)).result(timeout=0.5).data, b"2222")
                self.assertEqual(cache.snapshot()["l2_pending_bytes"], 4)
            finally:
                release.set()
            self.assertTrue(pending.result(timeout=2))
            self.settle(cache)
        self.assertEqual(cache.snapshot()["counters"].get("l2_queue_full", 0), 0)
        self.assertEqual(cache.snapshot()["counters"]["l2_backpressure_waits"], 1)
        cache.close()
        cache = self.cache(ram=16)
        self.assertEqual(cache.get(key(2)).data, b"2222")

    def test_backpressure_timeout_skips_l2_but_keeps_l1(self):
        cache = self.cache(ram=16, queue=4, wait=0.05)
        entered, release, gate = self.block_writer(cache)
        with gate:
            try:
                cache.put(meta(1, b"1111"), b"1111")
                self.assertTrue(entered.wait(2))
                started = time.monotonic()
                self.assertTrue(cache.put(meta(2, b"2222"), b"2222"))
                self.assertGreaterEqual(time.monotonic() - started, 0.04)
                self.assertEqual(cache.get(key(2)).data, b"2222")
                counters = cache.snapshot()["counters"]
                self.assertEqual(counters["l2_backpressure_timeouts"], 1)
                self.assertEqual(counters["l2_queue_full"], 1)
            finally:
                release.set()
            self.settle(cache)
        cache.close()
        cache = self.cache()
        self.assertIsNone(cache.get(key(2)))

    def test_remove_cancels_waiting_put_without_resurrection(self):
        cache = self.cache(ram=16, queue=4)
        entered, release, gate = self.block_writer(cache)
        with gate, ThreadPoolExecutor(max_workers=1) as executor:
            try:
                cache.put(meta(1, b"1111"), b"1111")
                self.assertTrue(entered.wait(2))
                pending = executor.submit(cache.put, meta(2, b"2222"), b"2222")
                self.wait_for_waiter(cache)
                cache.remove(key(2))
                self.assertFalse(pending.result(timeout=0.5))
                self.assertFalse(cache.contains(key(2)))
            finally:
                release.set()
            self.settle(cache)
        cache.close()
        cache = self.cache()
        self.assertIsNone(cache.get(key(2)))

    def test_close_wakes_waiting_put_and_drains_active_write(self):
        cache = self.cache(ram=16, queue=4)
        entered, release, gate = self.block_writer(cache)
        with gate, ThreadPoolExecutor(max_workers=2) as executor:
            try:
                cache.put(meta(1, b"1111"), b"1111")
                self.assertTrue(entered.wait(2))
                pending = executor.submit(cache.put, meta(2, b"2222"), b"2222")
                self.wait_for_waiter(cache)
                closed = executor.submit(cache.close)
                self.assertFalse(pending.result(timeout=0.5))
                self.assertFalse(closed.done())
                self.assertEqual(cache.snapshot()["counters"]["l2_wait_cancelled"], 1)
            finally:
                release.set()
            closed.result(timeout=5)
        cache = self.cache()
        self.assertEqual(cache.get(key(1)).data, b"1111")
        self.assertFalse(cache.contains(key(2)))


if __name__ == "__main__":
    unittest.main()
