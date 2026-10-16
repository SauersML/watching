"""Lossless, checksum-verified checkpoint transport against an existing base."""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

from watching.artifacts import write_json

BLOCK_SIZE = 1 << 20


def xor_bytes(a, b):
    if len(a) != len(b):
        raise ValueError("Checkpoint lengths differ")
    return (int.from_bytes(a, "little") ^ int.from_bytes(b, "little")).to_bytes(len(a), "little")


def encode(base: Path, target: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Delta output must be empty")
    base_hash, target_hash = hashlib.sha256(), hashlib.sha256()
    size = 0
    temporary = output / "weights.xor.gz.partial"
    with (
        base.open("rb") as a,
        target.open("rb") as b,
        gzip.open(temporary, "wb", compresslevel=1) as delta,
    ):
        while True:
            x, y = a.read(BLOCK_SIZE), b.read(BLOCK_SIZE)
            if not x and not y:
                break
            delta.write(xor_bytes(x, y))
            base_hash.update(x)
            target_hash.update(y)
            size += len(y)
    temporary.replace(output / "weights.xor.gz")
    write_json(
        output / "delta.json",
        dict(
            base_sha256=base_hash.hexdigest(),
            target_sha256=target_hash.hexdigest(),
            bytes=size,
            format="gzip-xor-v1",
        ),
    )


def decode(base: Path, delta_dir: Path, target: Path):
    if target.exists():
        raise FileExistsError(target)
    metadata = json.loads((delta_dir / "delta.json").read_text())
    if metadata["format"] != "gzip-xor-v1":
        raise ValueError("Unknown delta format")
    base_hash, target_hash = hashlib.sha256(), hashlib.sha256()
    size = 0
    temporary = target.with_name(target.name + ".partial")
    try:
        with (
            base.open("rb") as source,
            gzip.open(delta_dir / "weights.xor.gz", "rb") as delta,
            temporary.open("wb") as output,
        ):
            while True:
                x, difference = source.read(BLOCK_SIZE), delta.read(BLOCK_SIZE)
                if not x and not difference:
                    break
                restored = xor_bytes(x, difference)
                output.write(restored)
                base_hash.update(x)
                target_hash.update(restored)
                size += len(restored)
        if (
            size != metadata["bytes"]
            or base_hash.hexdigest() != metadata["base_sha256"]
            or target_hash.hexdigest() != metadata["target_sha256"]
        ):
            raise ValueError("Checkpoint checksum mismatch")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("encode", "decode"))
    parser.add_argument("base", type=Path, help="Base model.safetensors")
    parser.add_argument("checkpoint", type=Path, help="Target checkpoint or delta directory")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action == "encode":
        if args.out is None:
            parser.error("Encoding requires --out")
        encode(args.base, args.checkpoint / "model.safetensors", args.out)
        for path in args.checkpoint.iterdir():
            if path.is_file() and path.name != "model.safetensors":
                shutil.copy2(path, args.out / path.name)
    else:
        if args.out is not None:
            parser.error("Decoding writes model.safetensors inside the delta directory")
        decode(args.base, args.checkpoint, args.checkpoint / "model.safetensors")


if __name__ == "__main__":
    main()
