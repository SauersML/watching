"""Verify archived native answer scores and recompute control logit comparisons."""

import argparse
import json
from pathlib import Path

import torch

from watching.compile_control import answer_readout
from watching.tail_train import difference

CONTRASTS = (
    ("none_a", "none_b"),
    ("none_a", "buffer"),
    ("none_b", "buffer"),
    ("reference", "identity"),
    ("identity", "identity_reference"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--negative-id", type=int, default=2753)
    parser.add_argument("--positive-id", type=int, default=9454)
    args = parser.parse_args()
    report = {}
    for filename, readouts in (
        ("logits.pt", "base_answer_readouts.json"),
        ("trained_logits.pt", "trained_answer_readouts.json"),
    ):
        values = torch.load(args.raw / filename, map_location="cpu", weights_only=True)
        expected = json.loads((args.raw / readouts).read_text())
        actual = {
            name: [answer_readout(x, args.negative_id, args.positive_id) for x in rows]
            for name, rows in values.items()
        }
        for name, rows in actual.items():
            for observed, saved in zip(rows, expected[name], strict=True):
                for key, value in observed.items():
                    if key.endswith("_probability"):
                        # exp implementations can differ by one FP32 ULP across CPU builds.
                        original = torch.tensor(saved[key], dtype=torch.float32)
                        ulp = float(
                            torch.nextafter(original, torch.tensor(float("inf"))) - original
                        )
                        if abs(value - saved[key]) > ulp:
                            raise ValueError(f"Probability exceeds one FP32 ULP: {name}/{key}")
                    elif value != saved[key]:
                        raise ValueError(f"Archived native log score disagrees: {name}/{key}")
        report[filename] = {
            f"{left}__{right}": [
                difference(a, b) for a, b in zip(values[left], values[right], strict=True)
            ]
            for left, right in CONTRASTS
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Native log scores match exactly; probabilities match within one FP32 exp ULP.")


if __name__ == "__main__":
    main()
