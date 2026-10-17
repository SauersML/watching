"""Describe existing numerical footprints across user- and system-prefix formats.

No model execution, training, or causal intervention. Only saved train/development
states are inspected. Hidden-state differences do not establish learnability.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from watching.execution_fit import tensor_digest

ROLE_IDS = {872: "user", 8948: "system"}


def distribution(values):
    a = np.asarray(values, dtype=float)
    return {
        "min": float(a.min()),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "max": float(a.max()),
    }


def cosine(a, b):
    norm = float(a.norm() * b.norm())
    return None if norm == 0 else float(torch.dot(a, b) / norm)


def first_user_summary(rows):
    tokens = [r["user_role_occurrences"][0] for r in rows if r["user_role_occurrences"]]
    if not tokens:
        return None
    return {
        "n_prompts": len(tokens),
        "unique_absent_states": len({r["absent_sha256"] for r in tokens}),
        "unique_present_states": len({r["present_sha256"] for r in tokens}),
        "logical_position_range": [
            min(r["logical_position"] for r in tokens),
            max(r["logical_position"] for r in tokens),
        ],
        "delta_l2": distribution([r["delta_l2"] for r in tokens]),
    }


def analyze(cache):
    references, rows, role_details = {}, [], defaultdict(list)
    for split, pairs in cache["states"].items():
        if split not in ("train", "validation"):
            continue
        records = cache["data"][split]
        for index, pair in enumerate(pairs):
            record, batch = records[index], cache["batches"][split][index]
            valid = batch["attention_mask"][0].bool().nonzero().flatten()
            ids = batch["input_ids"][0, valid]
            pos = batch["position_ids"][0, valid]
            if (
                len(valid) < 6
                or int(ids[0]) != 151644
                or int(ids[1]) not in ROLE_IDS
                or int(ids[2]) != 198
                or not torch.equal(pos, torch.arange(len(valid)))
            ):
                raise ValueError("Unexpected chat prefix or logical positions")
            role = ROLE_IDS[int(ids[1])]
            if (role == "system") != record["group"].startswith("system_"):
                raise ValueError("Tokenized initial role disagrees with record format")
            absent, present = pair[0][0][0, valid], pair[1][0][0, valid]
            if absent.dtype != torch.bfloat16 or absent.shape != present.shape:
                raise ValueError("Expected matching native BF16 residual states")
            for a, b in zip(pair[0][1:], pair[1][1:], strict=True):
                if (a is None) != (b is None) or (a is not None and not torch.equal(a, b)):
                    raise ValueError("Auxiliary mask/rotary inputs differ across conditions")
            delta = present.double() - absent.double()
            midpoint = (present.double() + absent.double()) / 2
            role_state = (absent[1], present[1], delta[1])
            if role not in references:
                references[role] = tuple(t.clone() for t in role_state)
            reference = references[role]
            token_rows = []
            for logical in range(6):
                token_rows.append(
                    {
                        "logical_position": logical,
                        "physical_position": int(valid[logical]),
                        "token_id": int(ids[logical]),
                        "delta_l2": float(delta[logical].norm()),
                        "coordinates_different": int(torch.count_nonzero(delta[logical])),
                        "absent_sha256": tensor_digest(absent[logical]),
                        "present_sha256": tensor_digest(present[logical]),
                    }
                )
            user_roles = [
                int(i + 1)
                for i in range(len(ids) - 1)
                if int(ids[i]) == 151644 and int(ids[i + 1]) == 872
            ]
            row = {
                "split": split,
                "index": index,
                "context_id": record["context_id"],
                "base_context_id": record.get("base_context_id"),
                "group": record["group"],
                "task": record["task"],
                "initial_role": role,
                "n_active_tokens": len(valid),
                "first_six": token_rows,
                "role_absent_matches_first": torch.equal(role_state[0], reference[0]),
                "role_present_matches_first": torch.equal(role_state[1], reference[1]),
                "role_delta_matches_first": torch.equal(role_state[2], reference[2]),
                "role_delta_cosine_with_first": cosine(delta[1], reference[2]),
                "active_sequence_delta_l2": float(delta.norm()),
                "active_sequence_rms_coordinate_delta": float(delta.square().mean().sqrt()),
                "active_sequence_fraction_different": float((delta != 0).double().mean()),
                "answer_token_delta_l2": float(delta[-1].norm()),
                "answer_token_midpoint_l2": float(midpoint[-1].norm()),
                "user_role_occurrences": [
                    {
                        "logical_position": i,
                        "physical_position": int(valid[i]),
                        "delta_l2": float(delta[i].norm()),
                        "absent_sha256": tensor_digest(absent[i]),
                        "present_sha256": tensor_digest(present[i]),
                    }
                    for i in user_roles
                ],
            }
            rows.append(row)
            role_details[role].append(row)
    if set(references) != {"user", "system"}:
        raise ValueError("Need both initial role formats for this comparison")
    groups = {}
    for group in sorted({r["group"] for r in rows}):
        selected = [r for r in rows if r["group"] == group]
        groups[group] = {
            "n_prompts": len(selected),
            "first_actual_user_role": first_user_summary(selected),
            "splits": {s: sum(r["split"] == s for r in selected) for s in ("train", "validation")},
            **{
                key: distribution([r[key] for r in selected])
                for key in (
                    "active_sequence_delta_l2",
                    "active_sequence_rms_coordinate_delta",
                    "active_sequence_fraction_different",
                    "answer_token_delta_l2",
                )
            },
        }
    prefixes = {}
    for role, selected in role_details.items():
        prefixes[role] = {
            "n_prompts": len(selected),
            "all_role_absent_exactly_equal": all(r["role_absent_matches_first"] for r in selected),
            "all_role_present_exactly_equal": all(
                r["role_present_matches_first"] for r in selected
            ),
            "all_role_delta_exactly_equal": all(r["role_delta_matches_first"] for r in selected),
            "role_delta_l2": distribution([r["first_six"][1]["delta_l2"] for r in selected]),
            "first_six_by_position": [
                {
                    "logical_position": i,
                    "token_ids": sorted({r["first_six"][i]["token_id"] for r in selected}),
                    "unique_absent_states": len(
                        {r["first_six"][i]["absent_sha256"] for r in selected}
                    ),
                    "unique_present_states": len(
                        {r["first_six"][i]["present_sha256"] for r in selected}
                    ),
                    "delta_l2": distribution([r["first_six"][i]["delta_l2"] for r in selected]),
                }
                for i in range(6)
            ],
        }
    user, system = references["user"], references["system"]
    return {
        "groups": groups,
        "prefixes": prefixes,
        "rows": rows,
        "first_role_user_system_delta_cosine": cosine(user[2], system[2]),
        "first_role_user_system_midpoint_distance_l2": float(
            (
                ((user[0].double() + user[1].double()) - (system[0].double() + system[1].double()))
                / 2
            ).norm()
        ),
        "scope": "Descriptive saved source states at block34 entrance; no model forward, "
        "training, test-cohort analysis, or causal/learnability claim.",
    }


def main():
    from verify_tail_cache import sha256, verify

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-cache-sha256")
    args = parser.parse_args()
    torch.set_num_threads(4)
    path = args.source / "cache_states.pt"
    before = path.stat()
    digest = sha256(path)
    if args.expected_cache_sha256 and digest != args.expected_cache_sha256:
        raise ValueError("Cache checksum differs from requested snapshot")
    cache = torch.load(path, weights_only=True, map_location="cpu")
    files = {
        name: args.source / name
        for name in ("manifest.json", "data.json", "cache_fingerprints.json")
    }
    manifest, data, fingerprints = [json.loads(p.read_text()) for p in files.values()]
    # In-progress training caches contain only training/development states. Bind
    # each complete cached cohort to the corresponding archived data/fingerprints.
    selected = {s: data[s] for s in cache["data"]}
    selected_fingerprints = {s: fingerprints[s] for s in cache["data"]}
    count = verify(cache, selected, selected_fingerprints, manifest)
    report = analyze(cache)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Cache changed while being inspected; rerun on one stable snapshot")
    report["provenance"] = {
        "cache_sha256": digest,
        "cache_bytes": before.st_size,
        "all_cached_fingerprints_verified_prompt_pairs": count,
        "analyzed_prompt_pairs": len(report["rows"]),
        "source_sha256": {name: sha256(p) for name, p in files.items()},
        "source_model_revision": manifest.get("revision"),
        "cut_layer": cache["cut_layer"],
        "test_states_analyzed": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "prefixes")}, indent=2))
    print(
        json.dumps(
            {
                k: {key: value for key, value in v.items() if key != "first_six_by_position"}
                for k, v in report["prefixes"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
