"""Apply the verified DCP4 index-writer fix to an SM120 AOT catalog."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


# Server commit b4bc0c02fd01d08ce6175a4fbe0eedfd5f349467 removed these pre-patch variants.
STALE_HASHES = {
    "d1a2e2df1cffb0e847895b914044646fb22ad64bc0fd56ffe983f6934dc47e0d",
    "c0edad21f4cd12cc0b85f64fbcd9d85f3297ce93cdabee5482089fe62c2173ae",
    "b2252fe0bd5df7b19ee3067315327c26080d50989a70d77f3c6e6f84c3df4fe7",
    "c9b51512bf209d9fbe367b705a5cc0bec7188cdad68088b1e7de1a65094d1575",
}


def checked_catalog(path: Path, *, repair: bool = False) -> int:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    writers = {}
    stale = []
    for kernel in catalog["kernels"]:
        consts = kernel["consts"]
        if (kernel["fn"] != "_ik_write" or consts.get("DCP") != {"int": 4}
                or consts.get("ROWS") != {"int": 1} or kernel["num_warps"] != 4):
            continue
        # TensorFold matches parameters by name, not their order in the catalog.
        params = sorted(kernel["params"], key=lambda param: param["name"])
        signature = json.dumps([consts, params], sort_keys=True)
        if kernel["hash"] in STALE_HASHES:
            stale.append((signature, kernel))
            continue
        previous = writers.setdefault(signature, kernel)
        if previous["hash"] != kernel["hash"]:
            rank = consts["RANK"]["int"]
            raise ValueError(f"Ambiguous DCP4 _ik_write ROWS variants for rank {rank} in {path}")

    if not stale:
        return 0
    if not repair:
        raise ValueError(f"Stale DCP4 _ik_write ROWS variants in {path}. Repair or rebuild the AOT pack.")
    if any(signature not in writers for signature, _ in stale):
        raise ValueError(f"No replacement for a stale DCP4 _ik_write variant in {path}. Rebuild the AOT pack.")

    removed = {kernel["hash"] for _, kernel in stale}
    catalog["kernels"] = [kernel for kernel in catalog["kernels"] if kernel["hash"] not in removed]
    output = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    temporary = Path(output.name)
    try:
        with output:
            json.dump(catalog, output, indent=1)
            output.write("\n")
        temporary.chmod(path.stat().st_mode & 0o777)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return len(removed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--repair", action="store_true", help="remove verified stale entries; preserve the cubins")
    args = parser.parse_args()
    try:
        count = checked_catalog(args.catalog, repair=args.repair)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    if count:
        print(f"Removed {count} stale DCP4 _ik_write entries from {args.catalog}")
