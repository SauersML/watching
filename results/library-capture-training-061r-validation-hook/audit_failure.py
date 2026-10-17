"""Reproduce the failed fresh-process parity gate from persisted CPU native tensors."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from watching.execution_fit import tensor_digest


def audit(root):
    raw = root / "raw/evaluation"
    v = torch.load(raw / "extraction_failure.pt", map_location="cpu", weights_only=True)
    a, b = v["original_logits"], v["extracted_logits"]
    assert a.shape == b.shape and a.dtype == b.dtype
    d = b.float() - a.float()
    inputs = json.loads((raw / "inputs.json").read_text())["validation"][0]
    assert all(x.tolist() == inputs[k] for k, x in v["input"].items())
    manifest = json.loads((raw / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert int((d != 0).sum()) == v["report"]["different_logits"]
    assert float(d.abs().max()) == v["report"]["max_abs_difference"]
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return {
        "stage": v["report"]["stage"],
        "n_scored_prompt_pairs": 0,
        "input_matches_selected_validation_prompt_0": True,
        "original_logits_sha256": tensor_digest(a),
        "disabled_hook_logits_sha256": tensor_digest(b),
        "different_full_sequence_logits": int((d != 0).sum()),
        "total_full_sequence_logits": a.numel(),
        "max_full_sequence_abs_difference": float(d.abs().max()),
        "different_final_position_logits": int((d[0, -1] != 0).sum()),
        "max_final_position_abs_difference": float(d[0, -1].abs().max()),
        "bare_top_id": int(a[0, -1].argmax()),
        "disabled_hook_top_id": int(b[0, -1].argmax()),
        "bare_yes_no_log_odds": float(a.float()[0, -1, 9454] - a.float()[0, -1, 2753]),
        "disabled_hook_yes_no_log_odds": float(b.float()[0, -1, 9454] - b.float()[0, -1, 2753]),
        "raw_failure_file_sha256": hashlib.file_digest(
            (raw / "extraction_failure.pt").open("rb"), "sha256"
        ).hexdigest(),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).parent)
    args = p.parse_args()
    result = audit(args.root)
    recorded = json.loads((args.root / "independent_failure_review.json").read_text())
    assert all(recorded[k] == value for k, value in result.items())
    print(json.dumps({"status": "passed", "review_matches_persisted_tensors": True}))
