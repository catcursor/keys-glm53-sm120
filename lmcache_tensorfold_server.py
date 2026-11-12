"""Run LMCache's loopback remote server with a bounded CPU LRU."""

import argparse

from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.server.__main__ import LMCacheServer
from lmcache.v1.server.storage_backend.local_backend import LMSLocalBackend
from lmcache.v1.server.utils import LMSMemoryObj

CHUNK_BYTES = 4 << 20


class BoundedCache(LMSLocalBackend):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.used = 0

    def put(self, meta, data):
        if (
            data is None
            or len(data) != meta.length
            or not 0 < meta.length <= min(CHUNK_BYTES, self.capacity)
            or meta.fmt != MemoryFormat.BINARY_BUFFER
            or not meta.key.model_name.startswith("tf-glm53-v1-")
        ):
            return
        with self.lock:
            old = self.dict.pop(meta.key, None)
            if old is not None:
                self.used -= old.length
            while self.used + meta.length > self.capacity:
                _, evicted = self.dict.popitem(last=False)
                self.used -= evicted.length
            self.dict[meta.key] = LMSMemoryObj(data, meta.length, meta.fmt, meta.dtype, meta.shape)
            self.used += meta.length

    def get(self, key):
        with self.lock:
            obj = self.dict.get(key)
            if obj is not None:
                self.dict.move_to_end(key)
            return obj

    def remove(self, key):
        with self.lock:
            obj = self.dict.pop(key, None)
            if obj is not None:
                self.used -= obj.length


class TensorFoldCacheServer(LMCacheServer):
    def receive_all(self, sock, n):
        if not 0 < n <= CHUNK_BYTES:
            raise ValueError("Invalid LMCache frame size")
        return super().receive_all(sock, n)

    def handle_client(self, sock):
        sock.settimeout(10)
        super().handle_client(sock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gib", type=int, default=16, choices=range(1, 65))
    args = parser.parse_args()
    if not 0 < args.port < 65536:
        parser.error("--port must be between 1 and 65535")
    server = TensorFoldCacheServer("127.0.0.1", args.port, "cpu")
    server.data_store = BoundedCache(args.gib << 30)
    print(f"TensorFold LMCache listening on 127.0.0.1:{args.port} with a {args.gib} GiB CPU LRU", flush=True)
    server.run()


if __name__ == "__main__":
    main()
