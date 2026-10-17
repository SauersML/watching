"""Check whether a saved numerical signature is shared across chat prefixes.

This reads existing states; it does not run, train, or intervene on a model.
Exact equalities concern this finite cache, not unseen prompt formats or runtimes.
"""

import argparse
import json
from pathlib import Path

import torch

from watching.artifacts import write_json
from watching.execution_fit import tensor_digest
from watching.tail_train import file_digest

PREFIX = ((151644, "<|im_start|>"), (872, "user"), (198, "newline"))


def summarize_prefix(cache):
    references = {}
    rows = []
    groups = {name: [] for _, name in PREFIX}
    for split, records in cache["data"].items():
        for index in range(len(records)):
            batch = cache["batches"][split][index]
            pair = cache["states"][split][index]
            valid = batch["attention_mask"][0].bool().nonzero().flatten()
            if len(valid) < len(PREFIX):
                raise ValueError("Prompt has fewer tokens than the expected chat prefix")
            for logical_position, (token_id, name) in enumerate(PREFIX):
                position = int(valid[logical_position])
                if (
                    int(batch["input_ids"][0, position]) != token_id
                    or int(batch["position_ids"][0, position]) != logical_position
                ):
                    raise ValueError("Token or logical position differs from the expected prefix")
                absent, present = [state[0][0, position] for state in pair]
                if absent.dtype != torch.bfloat16 or present.dtype != absent.dtype:
                    raise ValueError("Expected native BF16 states")
                # FP64 subtraction and norms avoid adding low-precision analysis error.
                delta = present.double() - absent.double()
                midpoint = (present.double() + absent.double()) / 2
                if name not in references:
                    references[name] = (absent.clone(), present.clone(), delta.clone())
                ref_a, ref_p, ref_d = references[name]
                row = {
                    "split": split,
                    "index": index,
                    "prefix_token": name,
                    "token_id": token_id,
                    "physical_position": position,
                    "logical_position": logical_position,
                    "absent_sha256": tensor_digest(absent),
                    "present_sha256": tensor_digest(present),
                    "delta_l2": float(delta.norm()),
                    "midpoint_l2": float(midpoint.norm()),
                    "coordinates_different": int(torch.count_nonzero(delta)),
                    "absent_equals_first": torch.equal(absent, ref_a),
                    "present_equals_first": torch.equal(present, ref_p),
                    "delta_equals_first": torch.equal(delta, ref_d),
                }
                rows.append(row)
                groups[name].append(row)
    if not rows:
        raise ValueError("Empty source cache")
    summary = {}
    for name, items in groups.items():
        summary[name] = {
            "prompt_count": len(items),
            "unique_absent_hashes": len({r["absent_sha256"] for r in items}),
            "unique_present_hashes": len({r["present_sha256"] for r in items}),
            "all_absent_exactly_equal": all(r["absent_equals_first"] for r in items),
            "all_present_exactly_equal": all(r["present_equals_first"] for r in items),
            "all_deltas_exactly_equal": all(r["delta_equals_first"] for r in items),
            "nonzero_delta_prompts": sum(r["coordinates_different"] > 0 for r in items),
            "delta_l2_range": [
                min(r["delta_l2"] for r in items),
                max(r["delta_l2"] for r in items),
            ],
            "physical_position_range": [
                min(r["physical_position"] for r in items),
                max(r["physical_position"] for r in items),
            ],
        }
    return {"summary": summary, "rows": rows}


def main():
    from verify_tail_cache import verify

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cache_path = args.source / "cache_states.pt"
    actual_digest = file_digest(cache_path)
    if actual_digest != args.cache_sha256:
        raise ValueError("Source cache checksum mismatch")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    manifest = json.loads((args.source / "manifest.json").read_text())
    data = json.loads((args.source / "data.json").read_text())
    fingerprints = json.loads((args.source / "cache_fingerprints.json").read_text())
    verify(cache, data, fingerprints, manifest)
    result = summarize_prefix(cache)
    result["provenance"] = {
        "cache_sha256": actual_digest,
        "source_manifest_sha256": file_digest(args.source / "manifest.json"),
        "cut_layer": cache["cut_layer"],
        "cohort_sizes": {split: len(records) for split, records in data.items()},
        "all_cached_tensor_fingerprints_verified": True,
        "interpretation": "Descriptive existing-cache analysis; no new causal or population test",
    }
    write_json(args.out, result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
