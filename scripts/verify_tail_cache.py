"""Verify a completed native-tail cache against archived data and tensor fingerprints."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from watching.execution_fit import tensor_digest
from watching.null_tail import validate_tokenized_records
from watching.tail_train import context_key, state_fingerprint


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(cache, data, fingerprints, manifest):
    if cache.get("version") != 1 or cache.get("conditions") != [False, True]:
        raise ValueError("Unexpected cache version or condition order")
    if cache.get("data") != data or cache.get("checkpoint") != manifest["checkpoint"]:
        raise ValueError("Cache dataset or model differs from the archive")
    if cache.get("length") != manifest["length"]:
        raise ValueError("Cache sequence length differs")
    if any(
        set(mapping) != set(data) for mapping in [cache["states"], cache["batches"], fingerprints]
    ):
        raise ValueError("Cache/fingerprint split identities differ")
    count = 0
    for split, records in data.items():
        states, batches, digests = (
            cache["states"][split],
            cache["batches"][split],
            fingerprints[split],
        )
        if not len(records) == len(states) == len(batches) == len(digests):
            raise ValueError("Cache record counts differ")
        for index, (record, pair, batch, digest) in enumerate(
            zip(records, states, batches, digests, strict=True)
        ):
            if digest["index"] != index or digest["context_id"] != context_key(record):
                raise ValueError("Cache context identity differs")
            if len(pair) != 2 or any(len(state) != 4 for state in pair):
                raise ValueError("Cache requires two four-tensor conditions")
            positions = batch["position_ids"]
            expected = (batch["attention_mask"].long().cumsum(-1) - 1).clamp_min(0)
            if not torch.equal(positions, expected) or tuple(positions.shape) != (
                1,
                manifest["length"],
            ):
                raise ValueError("Position IDs differ from the attention mask or shape")
            if tensor_digest(positions) != digest["positions_sha256"]:
                raise ValueError("Position fingerprint differs")
            for name, state in zip(["absent", "present"], pair, strict=True):
                if state_fingerprint(state) != digest[name]:
                    raise ValueError("Tensor fingerprint differs")
                if any(
                    t is not None and (t.device.type != "cpu" or not torch.isfinite(t).all())
                    for t in state
                ):
                    raise ValueError("Nonfinite or non-CPU cache state")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True, help="Local cached tokenizer; no downloads")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest, data, fingerprints = [
        json.loads((args.fit / name).read_text())
        for name in ["manifest.json", "data.json", "cache_fingerprints.json"]
    ]
    before = args.cache.stat()
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    count = verify(cache, data, fingerprints, manifest)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, padding_side="left"
    )
    validate_tokenized_records(tokenizer, data, cache["batches"], manifest["length"])
    digest = sha256(args.cache)
    after = args.cache.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Cache changed during verification")
    report = {
        "cache_sha256": digest,
        "cache_bytes": after.st_size,
        "prompt_pairs_verified": count,
        "splits": {split: len(rows) for split, rows in data.items()},
        "data_exactly_equal_to_saved_cache_metadata": True,
        "every_state_shape_stride_dtype_sha256_matches": True,
        "every_position_id_matches_mask_and_sha256": True,
        "every_saved_input_id_and_attention_mask_matches_tokenized_text": True,
        "all_cached_tensors_finite_and_cpu": True,
        "source_files_sha256": {
            name: sha256(args.fit / name)
            for name in ["manifest.json", "data.json", "cache_fingerprints.json"]
        },
        "scope": "Artifact consistency only; no model execution or performance claims.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
