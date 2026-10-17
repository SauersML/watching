"""CPU positional comparison of the two released bare compilations."""

import argparse
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def review(directory):
    a_path = directory / "bare_omitted_1.safetensors"
    b_path = directory / "bare_omitted_2.safetensors"
    a, b = load_file(a_path)["logits"], load_file(b_path)["logits"]
    inputs = json.loads((directory / "inputs.json").read_text())
    mask = inputs["attention_mask"][0]
    difference = (a.float() - b.float()).abs()
    rows = [
        {
            "position": i,
            "attention_mask": mask[i],
            "input_token": inputs["input_ids"][0][i],
            "different_logits": int((a[0, i] != b[0, i]).sum()),
            "max_abs_difference": float(difference[0, i].max()),
            "argmax_a": int(a[0, i].argmax()),
            "argmax_b": int(b[0, i].argmax()),
        }
        for i in range(a.shape[1])
    ]
    return {
        "source_sha256": {p.name: sha(p) for p in [a_path, b_path, directory / "inputs.json"]},
        "script_sha256": sha(Path(__file__)),
        "first_differing_position": next(r["position"] for r in rows if r["different_logits"]),
        "mask_ones": sum(mask),
        "different_argmax_positions": [
            r["position"] for r in rows if r["argmax_a"] != r["argmax_b"]
        ],
        "positions": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(review(args.directory), indent=2) + "\n")
