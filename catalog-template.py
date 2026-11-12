"""Convert the published catalog metadata into a Triton sweep template.

Only parameter metadata crosses architectures. The sweep emits fresh SM120 cubins.
"""

import json
import struct
import sys
from pathlib import Path


def constexpr(value):
    if "int" in value:
        return value["int"]
    if "none" in value:
        return None
    if "f32" in value:
        bits = value["f32"]
        number = struct.unpack("<f", struct.pack("<I", bits))[0]
        fp64 = struct.unpack("<Q", struct.pack("<d", number))[0]
        return {"fp32_bits": f"{bits:08x}", "fp64_bits": f"{fp64:016x}"}
    raise ValueError(f"Unknown catalog constant: {value}")


if __name__ == "__main__":
    source, destination = (Path(name) for name in sys.argv[1:])
    variants = json.loads(source.read_text())["kernels"]
    kernels = [
        {
            "name": variant["fn"],
            "function": variant["fn"],
            "constexprs": {key: constexpr(value) for key, value in variant["consts"].items()},
            "metadata": {"num_stages": variant["num_stages"]},
        }
        for variant in variants
    ]
    destination.write_text(json.dumps({"kernels": kernels}))
