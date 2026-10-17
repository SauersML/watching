"""Describe fixed-endpoint scores separately for equal and unequal learner inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from watching.metrics import summarize_pairs
from watching.viewing_train import learner_input_equality


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def subset_report(rows, flags, seed):
    pairs = {}
    for row in rows:
        index, condition = row["index"], row["capture"]
        if index not in range(len(flags)) or type(condition) is not bool:
            raise ValueError("Score identity is outside the saved source pairs")
        pair = pairs.setdefault(index, {})
        if condition in pair:
            raise ValueError("Duplicate scored condition")
        pair[condition] = row
    if set(pairs) != set(range(len(flags))) or any(set(p) != {False, True} for p in pairs.values()):
        raise ValueError("Every saved source pair must have both scored conditions")
    for index, equal in enumerate(flags):
        if equal:
            a, b = pairs[index][False], pairs[index][True]
            if any(a[key] != b[key] for key in ("logp_yes", "logp_no", "yes_minus_no")):
                raise ValueError(
                    "Identical learner inputs have unequal scored answer probabilities"
                )
            if a["greedy_correct"] and b["greedy_correct"]:
                raise ValueError("Identical inputs cannot both have distinct correct top tokens")
    output = {}
    for name, equal in (("identical_states", True), ("different_states", False)):
        selected = [row for row in rows if flags[row["index"]] == equal]
        output[name] = (
            summarize_pairs(selected, seed=seed, bootstrap_samples=5000) if selected else None
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.fit / "manifest.json").read_text())
    step = manifest["completed_steps"]
    if manifest.get("training_status") != "complete" or step != manifest["steps"]:
        raise ValueError("Only the prespecified completed endpoint may be analyzed")
    cache_path = args.fit / "cache_states.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    data = json.loads((args.fit / "data.json").read_text())
    torch.set_num_threads(4)
    report = {
        "completed_steps": step,
        "viewing_mode": manifest["viewing_mode"],
        "cache_sha256": digest(cache_path),
        "data_sha256": digest(args.fit / "data.json"),
        "scope": "Post-run descriptive analysis of fixed training/development pairs; no test "
        "selection or refitting. Equality includes hidden state, mask, cosine and sine bytes; "
        "both conditions share position IDs. Intervals resample whole prompts within each "
        "pre-existing input-equality subgroup, conditional on one training/compilation run.",
        "splits": {},
    }
    for split in ("train", "validation"):
        if cache["data"][split] != data[split]:
            raise ValueError("Cache records differ from scored run data")
        flags = [
            learner_input_equality({"train": [pair], "validation": [pair]})[
                "all_learner_states_bit_identical"
            ]
            for pair in cache["states"][split]
        ]
        if len(flags) != len(data[split]):
            raise ValueError("Cache pair count differs from records")
        points = {}
        for endpoint in (0, step):
            path = args.fit / f"{split}_{endpoint}_split_inference.json"
            scores = json.loads(path.read_text())
            if (
                scores["summary"]["path"] != "split_inference"
                or scores["summary"]["step"] != endpoint
            ):
                raise ValueError("Score runtime or checkpoint metadata mismatch")
            points[str(endpoint)] = {
                "source_sha256": digest(path),
                "subsets": subset_report(scores["rows"], flags, manifest["seed"]),
            }
        report["splits"][split] = {
            "pair_identity": [
                {
                    "index": i,
                    "context_id": record.get("context_id"),
                    "states_bit_identical": flags[i],
                }
                for i, record in enumerate(data[split])
            ],
            "endpoints": points,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
