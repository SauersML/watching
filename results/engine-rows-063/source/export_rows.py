"""Install two new output rows (Yes and No) into Qwen3-8B; every other tensor is the original.

The pinned checkpoint keeps lm_head.weight alone in its fifth shard, so only that shard is
rewritten; every other file is linked to the original snapshot.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SHARD = "model-00005-of-00005.safetensors"


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = load_file(str(args.rows))
    yes, no = int(rows["yes_id"]), int(rows["no_id"])
    args.out.mkdir(parents=True, exist_ok=False)
    for path in sorted(args.base.iterdir()):
        if path.name != SHARD:
            os.symlink(path.resolve(), args.out / path.name)
    with safe_open(str(args.base / SHARD), "pt") as handle:
        assert list(handle.keys()) == ["lm_head.weight"], handle.keys()
        metadata = handle.metadata()
        original = handle.get_tensor("lm_head.weight")
    patched = original.clone()
    patched[yes] = rows["yes_row"].to(torch.bfloat16)
    patched[no] = rows["no_row"].to(torch.bfloat16)
    save_file({"lm_head.weight": patched}, str(args.out / SHARD), metadata=metadata)
    reloaded = load_file(str(args.out / SHARD))["lm_head.weight"]
    changed = (reloaded != original).any(-1).nonzero().flatten().tolist()
    assert set(changed) <= {yes, no} and torch.equal(reloaded, patched), changed
    report = {
        "yes_id": yes,
        "no_id": no,
        "changed_rows": changed,
        "rows_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(),
        "original_lm_head_sha256": digest(original),
        "patched_lm_head_sha256": digest(reloaded),
        "other_files": "symlinks to the pinned snapshot",
    }
    (args.out.parent / "export.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
