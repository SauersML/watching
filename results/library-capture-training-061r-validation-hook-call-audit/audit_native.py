"""CPU audit of released full-sequence logits; no model/GPU/compilation."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def compare(left, right):
    difference = (left.float() - right.float()).abs()
    return {
        "bit_identical": digest(left) == digest(right),
        "different_logits": int((left != right).sum()),
        "max_abs_difference": float(difference.max()),
        "different_final_position_logits": int((left[:, -1] != right[:, -1]).sum()),
        "max_final_position_abs_difference": float(difference[:, -1].max()),
    }


def audit(evaluation):
    report = json.loads((evaluation / "report.json").read_text())
    vectors, arms = {}, {}
    for arm in report["arms"]:
        name = arm["name"]
        native = load_file(evaluation / (name + ".safetensors"))
        left, right = native["logits"], native["repeat_logits"]
        assert left.shape == right.shape == (1, 96, 151936)
        assert left.dtype == right.dtype == torch.bfloat16
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        assert digest(left) == arm["logits_sha256"]
        repeat = compare(left, right)
        assert repeat == arm["repeat"]
        ids = report["answer_ids"]
        gap = float(left.float()[0, -1, ids["Yes"]] - left.float()[0, -1, ids["No"]])
        assert gap == arm["final_yes_no_log_odds"]
        assert int(left[0, -1].argmax()) == arm["final_top_token_id"]
        arms[name] = {"sha256": digest(left), "repeat": repeat, "gap": gap}
        vectors[name] = left
    comparisons = {}
    for key, expected in report["comparisons"].items():
        left, right = key.split("_vs_")
        comparisons[key] = compare(vectors[left], vectors[right])
        assert comparisons[key] == expected
    assert report["state_before"] == report["state_after"]
    assert all(arm["single_graph_reused"] for arm in report["arms"])
    assert report["attribution_admissible"] is False
    return {
        "status": "verified",
        "shape": [1, 96, 151936],
        "dtype": "bfloat16",
        "arms": arms,
        "comparisons": comparisons,
        "state_unchanged": True,
        "attribution_admissible": False,
        "interpretation": (
            "Stable repeats within graphs; independent bare compilations differ. "
            "No signature/hook attribution or accuracy inference."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(audit(args.evaluation), indent=2) + "\n")
