"""Install the trained tail tensors into Qwen3-8B; every other tensor is the original.

Shards holding a trained tensor are rewritten; every other file is linked to the snapshot."""

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tail", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    trained = load_file(str(args.tail))
    index = json.loads((args.base / "model.safetensors.index.json").read_text())["weight_map"]
    missing = [n for n in trained if n not in index]
    assert not missing, missing
    shards = {index[n] for n in trained}
    args.out.mkdir(parents=True)
    changed = {}
    for item in args.base.iterdir():
        if item.name in shards:
            tensors = load_file(str(item))
            for name in [n for n in trained if index[n] == item.name]:
                assert tensors[name].shape == trained[name].shape and tensors[name].dtype == trained[name].dtype
                changed[name] = float((tensors[name].float() - trained[name].float()).norm())
                tensors[name] = trained[name]
            save_file(tensors, str(args.out / item.name), metadata={"format": "pt"})
        else:
            os.symlink(item.resolve(), args.out / item.name)
    (args.out.parent / "export.json").write_text(json.dumps({"changed_norms": changed, "shards": sorted(shards)}, indent=1) + "\n")
    print("EXPORT_COMPLETE", len(changed), sorted(shards), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
