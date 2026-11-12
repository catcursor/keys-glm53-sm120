"""Fetch pinned kernel sources, published launch metadata and the ft2 drafter."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
MIRROR = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
REPO = "drowzeys/keys-GLM-5.3-speculator.dspark-ft2"
REVISION = "681d43db532b2f004d21c1d5ebe4aef5678d564f"
PYTHON_SOURCE = "b6e3e8d16453a008f10450cd40b124d5b799b802"
IMAGE = "drowzeys/keys-tensorfold-glm53-tp4-dgx-spark"
MANIFEST = "sha256:6290c15f1596347468fa63c35f9e5ebc37212294fd283fc1c7165fddbcb9b34d"
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch(url, path, expected, *, size=None, headers=None, ranged=False):
    algorithm = "sha256" if len(expected) == 64 else "sha1"

    def matches(file):
        if not file.is_file() or (size is not None and file.stat().st_size != size):
            return False
        with file.open("rb") as stream:
            if algorithm == "sha256":
                return hashlib.file_digest(stream, algorithm).hexdigest() == expected
            digest = hashlib.sha1(f"blob {size}\0".encode())
            digest.update(stream.read())
            return digest.hexdigest() == expected

    if matches(path):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    offset = part.stat().st_size if size is not None and part.is_file() else 0
    if not matches(part):
        request = urllib.request.Request(url, headers=headers or {})
        if size is not None and offset < size and (ranged or offset):
            request.add_header("Range", f"bytes={offset}-")
        with DIRECT.open(request, timeout=60) as response:
            if response.status == 206:
                if response.headers.get("Content-Range") != f"bytes {offset}-{size - 1}/{size}":
                    raise RuntimeError(f"Unexpected content range for {path.name}")
                mode = "ab"
            elif response.status == 200:
                mode = "wb"
            else:
                raise RuntimeError(f"Unexpected status {response.status} for {path.name}")
            with part.open(mode) as output:
                shutil.copyfileobj(response, output, 2**20)
    if not matches(part):
        raise RuntimeError(f"Checksum mismatch for {path.name}")
    part.replace(path)
    print(f"Verified {path.name}", flush=True)


def read_json(url, headers=None):
    with DIRECT.open(urllib.request.Request(url, headers=headers or {}), timeout=30) as response:
        return json.load(response)


def prepare_drafter():
    model = read_json(f"{MIRROR}/api/models/{REPO}/revision/{REVISION}?blobs=true")
    if model.get("sha") != REVISION:
        raise RuntimeError("DSpark revision mismatch")
    files = {item["rfilename"]: item for item in model["siblings"]}
    for name in ("model.safetensors", "config.json", "LICENSE"):
        item = files[name]
        digest = (item.get("lfs") or {}).get("sha256") or item["blobId"]
        if name.endswith(".safetensors"):
            url = f"{MIRROR}/{REPO}/resolve/{REVISION}/{name}?download=true&request={time.time_ns()}"
        else:
            url = f"{MIRROR}/{REPO}/raw/{REVISION}/{name}"
        fetch(url, ROOT / "dspark-ft2" / name, digest, size=item["size"], ranged=name.endswith(".safetensors"))


def prepare_reference():
    path = ROOT / "python-reference"
    if not (path / "src/tensorfold/families/glm_moe_dsa/cuda/dspark.py").is_file():
        url = f"https://codeload.github.com/drowzeys/TensorFold/tar.gz/{PYTHON_SOURCE}"
        with DIRECT.open(url, timeout=60) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                if not member.isfile() or len(parts) < 3 or parts[1] != "src" or ".." in parts:
                    continue
                out = path.joinpath(*parts[1:])
                out.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, out.open("wb") as output:
                    shutil.copyfileobj(source, output)


def prepare_templates():
    scope = urllib.parse.urlencode({"service": "ghcr.io", "scope": f"repository:{IMAGE}:pull"})
    token = read_json(f"https://ghcr.io/token?{scope}")["token"]
    headers = {"Authorization": "Bearer " + token}
    api = f"https://ghcr.io/v2/{IMAGE}"
    manifest = ROOT / "registry/manifest.json"
    fetch(f"{api}/manifests/{MANIFEST}", manifest, MANIFEST.split(":")[1],
          headers={**headers, "Accept": "application/vnd.oci.image.manifest.v1+json"})
    wanted = {
        ("ref", "bmm_probe.json"), ("ref", "inv_freq.bin"),
        *(("ref", f"tiles-r{rank}.json") for rank in range(4)),
        ("aot", "aot.json"),
        ("aot-wide", "aot.json"),
    }
    extracted = set()
    # The first three layers are the base OS and the TensorFold Python environment.
    for layer in reversed(json.loads(manifest.read_text())["layers"][3:]):
        digest = layer["digest"].split(":")[1]
        blob = ROOT / "registry" / digest
        fetch(f"{api}/blobs/{layer['digest']}", blob, digest, size=layer["size"], headers=headers)
        with tarfile.open(blob) as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                if not member.isfile() or parts[:2] != ("opt", "keys-glm53"):
                    continue
                relative = parts[2:]
                if relative not in wanted or relative in extracted:
                    continue
                out = ROOT.joinpath(*relative)
                out.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, out.open("wb") as output:
                    shutil.copyfileobj(source, output)
                print(f"Extracted {member.name}", flush=True)
                extracted.add(relative)
        if extracted == wanted:
            (ROOT / "ref-dcp4").mkdir(exist_ok=True)
            for source, target in (
                ("aot/aot.json", "ref/manifest.json"),
                ("aot-wide/aot.json", "ref-dcp4/manifest.json"),
            ):
                subprocess.run([sys.executable, "-B", str(ROOT / "catalog-template.py"),
                                str(ROOT / source), str(ROOT / target)], check=True)
            return
    raise RuntimeError(f"Missing reference assets: {sorted(wanted - extracted)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("asset", choices=("reference", "templates", "drafter"))
    args = parser.parse_args()
    {"reference": prepare_reference, "templates": prepare_templates, "drafter": prepare_drafter}[args.asset]()
